import torch
import torch.nn as nn
import numpy as np
from src.utils.skeleton.graph import coarsen, coarsened_graph, adjacency2graph
from .util import to_tuple

class GraphConv(nn.Module):
    """
    Graph Convolution. The dim layout are (batch, channel, graph, spatial/temporal...).
    Examples:
    * (batch, channel, graph)
    * (batch, channel, graph, time)
    * (batch, channel, graph, height, width)

    Two very different use cases:
    1. static graph processing (will cache the computation kernel)
        A. precompute everything
            mode=manual, computation_kernel=kernel, in_graph=None, out_vertices=None, out_vertices_num=0, return_graph=False, dynamic=False
        B. pre-selected vertices, auto compute out graphs
            mode=vertex, computation_kernel=None, in_graph=graph, out_vertices=vertices, out_vertices_num=0, return_graph=True, dynamic=False
        C. predefined number of vertices, auto select vertices and compute out graphs
            mode=number, computation_kernel=None, in_graph=graph, out_vertices=None, out_vertices_num=number, return_graph=True, dynamic=False
        D. auto-compute everything (Default)
            mode=auto, computation_kernel=None, in_graph=graph, out_vertices=None, out_vertices_num=0, return_graph=True, dynamic=False
    2. dynamic graph processing (without cache)

    Args:
        extra_dim (int): Extra spatial/temporal dim for graph conv, supported values are 0, 1, 2
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the graph convolution
        kernel_size (int or tuple): Graph dim supports kernel size. The first element is graphical kernel size
        stride (int or tuple): For the graph dim, should be kernel >= stride, currently only stride 1,2 are supported
                               When stride = 1, only 'auto' and 'manual' modes are allowed
        padding (int or tuple): For the graph dim, should be 0
        dilation (int or tuple): Graph dim supports dilation
        groups (int): Graph dim supports group channels
        bias (bool): Bias
        spectral_norm (bool): Enable spectral norm
        mode: Mode of coarsen, auto | number | vertex | manual
        computation_kernel: What the computation actually uses
        in_graph (Tensor): Adjacency kernel of input graph
        out_vertices (None | list): Vertices selected by the graph convolution, default 0 for auto selection.
        out_vertices_num (int): How many selected vertices
        return_graph (bool): Whether return the adjacency matrix of out graph
        symmetry_break (bool): (none | random | radial) Whether break symmetry
        pooling (bool): use pooling instead of stride for non-graph dimensions
        dynamic (bool): Default False. Not implemented yet
    """

    def __init__(self,
                 extra_dim,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 spectral_norm=False,
                 mode='auto',
                 computation_kernel=None,
                 in_graph=None,
                 out_vertices=None,
                 out_vertices_num=0,
                 return_graph=True,
                 symmetry_break=False,
                 normalize='group',
                 pooling=False,
                 dynamic=False,
                 use_graph_conv=True,
                 use_node_bias=False):
        super().__init__()
        self.extra_dim = extra_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = to_tuple(kernel_size, extra_dim + 1)
        self.stride = to_tuple(stride, extra_dim + 1)
        self.padding = to_tuple(padding, extra_dim + 1)
        self.dilation = to_tuple(dilation, extra_dim + 1)
        self.groups = groups
        self.bias = bias
        self.spectral_norm = spectral_norm
        self.mode = mode
        self.computation_kernel = computation_kernel
        self.in_graph = in_graph
        self.out_vertices = out_vertices
        self.out_vertices_num = out_vertices_num
        self.return_graph = return_graph
        self.symmetry_break = symmetry_break
        self.normalize = normalize
        self.pooling = pooling
        self.dynamic = dynamic
        self.out_graph = None
        self.use_graph_conv = use_graph_conv
        self.use_node_bias = use_node_bias
        if self.stride[0] == 1:
            assert self.mode in ['auto', 'manual']
            self.out_graph = self.in_graph
            if self.in_graph is not None:
                self.out_vertices = list(range(self.in_graph.shape[0]))
                self.out_vertices_num = len(self.out_vertices)

        # sanity check
        if dynamic: raise NotImplementedError()
        assert extra_dim in [0, 1, 2]
        assert self.padding[0] == 0
        assert self.kernel_size[0] >= self.stride[0]
        assert self.stride[0] in [1, 2]
        assert normalize in ['group', 'joint', 'none']
        assert symmetry_break in ['none', 'radom', 'radial', True, False]
        if pooling: assert extra_dim > 0

        if mode == 'auto':
            assert computation_kernel is None
            assert in_graph is not None
            assert out_vertices is None
            assert out_vertices_num == 0
        elif mode == 'number':
            assert computation_kernel is None
            assert in_graph is not None
            assert out_vertices is None
            assert out_vertices_num > 0
        elif mode == 'vertex':
            assert computation_kernel is None
            assert in_graph is not None
            assert out_vertices is not None
            assert out_vertices_num == 0
        elif mode == 'manual':
            assert computation_kernel is not None
            assert in_graph is None
            assert out_vertices is None
            assert out_vertices_num == 0
            assert not return_graph
        else:
            raise ValueError(f'mode {mode} not supported')

        self.computation_kernel = self.get_computation_kernel()
        self.computation_kernel_size, self.out_vertices_num, self.in_vertices_num = self.computation_kernel.shape
        if self.return_graph: self.out_graph = self.get_out_graph()

        A = torch.tensor(self.computation_kernel, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        if extra_dim == 0:
            Conv = nn.Conv1d
        elif extra_dim == 1:
            Conv = nn.Conv2d
            Pool = nn.AvgPool2d
        elif extra_dim == 2:
            Conv = nn.Conv3d
            Pool = nn.AvgPool3d
        else:
            raise ValueError('extra dim must in 0, 1, 2')

        if use_graph_conv:
            num_node_kernel = 1
        else:
            num_node_kernel = self.in_vertices_num

        if pooling:
            self.conv = Conv(
                self.in_channels * num_node_kernel,
                self.out_channels * self.computation_kernel_size * num_node_kernel,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=self.groups * num_node_kernel,
                bias=self.bias)
            self.pool = Pool(
                kernel_size=(1, *(self.kernel_size[1:])),
                stride=(1, *(self.stride[1:])),
                padding=(0, *(self.padding[1:])))
        else:
            self.conv = Conv(
                self.in_channels * num_node_kernel,
                self.out_channels * self.computation_kernel_size * num_node_kernel,
                kernel_size=(1, *(self.kernel_size[1:])),
                stride=(1, *(self.stride[1:])),
                padding=(0, *(self.padding[1:])),
                dilation=(1, *(self.dilation[1:])),
                groups=self.groups * num_node_kernel,
                bias=self.bias)

        if spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        if use_node_bias:
            bias_data = torch.Tensor(
                self.out_channels, self.out_vertices_num)
            bias_data.fill_(0)
            self.node_bias = nn.Parameter(bias_data)
            
    def forward(self, x, graph=None):
        """
        Input shape: (batch, channel, graph, spatial/temporal)
                        n       c       v       ab  /   t

        Output shape: (batch, channel, graph, spatial/temporal)
                         n       c       w       ab  /   t
        """
        c, k, w = self.out_channels, self.computation_kernel_size, self.out_vertices_num

        if graph is not None: raise NotImplementedError()
        if self.use_graph_conv:
            x = self.conv(x)
        else:
            assert len(x.size()) == 4
            n, c_in, v, t = x.size()
            x = x.permute(0, 2, 1, 3).contiguous().view(n, v*c_in, 1, t)
            x = self.conv(x)
            x = x.view(n, v, c*k, x.size(3)).permute(0, 2, 1, 3)


        if self.extra_dim == 0:
            n, ck, v = x.size()
            assert ck == c * k
            x = x.view(n, c, k, v)
            x = torch.einsum('kwv,nckv->ncw', (self.A.data, x))
            assert x.size() == (n, c, w)
        elif self.extra_dim == 1:
            n, ck, v, t = x.size()
            assert ck == c * k
            x = x.view(n, c, k, v, t)
            x = torch.einsum('kwv,nckvt->ncwt', (self.A.data, x))
            assert x.size() == (n, c, w, t)
        elif self.extra_dim == 2:
            n, ck, v, a, b = x.size()
            assert ck == c * k
            x = x.view(n, c, k, v, a, b)
            x = torch.einsum('kwv,nckvab->ncwab', (self.A.data, x))
            assert x.size() == (n, c, w, a, b)
        x = x.contiguous()

        if self.pooling:
            x = self.pool(x)

        if self.use_node_bias:
            assert len(x.size()) == 4
            x = x + self.node_bias.unsqueeze(0).unsqueeze(3)


        if self.return_graph:
            assert self.out_graph is not None
            return x, self.out_graph

        return x

    def get_computation_kernel(self):
        if self.computation_kernel is not None:
            return self.computation_kernel
        assert self.in_graph is not None
        if self.out_vertices is not None:
            return conv_computation_kernel(self.in_graph, self.out_vertices, self.kernel_size[0], self.dilation[0],
                                           self.symmetry_break, self.normalize)
        assert self.stride[0] == 2
        if self.out_vertices_num > 0:
            assert self.mode == 'number'
            self.out_graph, self.out_vertices = coarsen(self.in_graph, self.out_vertices_num)
        else:
            assert self.mode == 'auto'
            self.out_graph, self.out_vertices = coarsen(self.in_graph)

        return self.get_computation_kernel()

    def get_out_graph(self):
        if self.out_graph is not None:
            return self.out_graph
        assert self.in_graph is not None
        assert self.out_vertices is not None
        return coarsened_graph(self.in_graph, self.out_vertices)


def conv_computation_kernel(in_adj, out_vertices, kernel, dilation, symmetry_break, normalize='group'):
    """
    Get the computation kernel of graph convolution.

    Shape of the kernel: K * W * V
    K: ((kernel - 1) * dilation + 1) * 2 ^ (symmetry_break) - (symmetry_break)
    W: num out vertex
    V: num in vertex
    """
    V = in_adj.shape[0]
    W = len(out_vertices)
    K = (kernel - 1) * dilation + 1
    assert W == len(out_vertices)
    adj = np.zeros((K, V, V))
    g = adjacency2graph(in_adj)
    for i in range(V):
        ps = g.get_all_shortest_paths(i)
        for k in range(0, K, dilation):
            for j, p in enumerate(ps):
                if len(p) == k + 1:
                    adj[k, i, j] = 1
    assert (adj == np.transpose(adj, (0, 2, 1))).all()
    adj = adj[:, out_vertices, :]
    assert not np.isnan(np.min(adj))

    if normalize == 'group':
        adj = _group_normalize(adj)
    elif normalize == 'joint':
        adj = _joint_normalize(adj)
    else:
        assert normalize == 'none'

    if symmetry_break in [True, 'random']:
        adj = _symmbreak(adj, out_vertices)
    elif symmetry_break == 'radial':
        adj = _symmbreak_radial(adj, out_vertices, g)

    return adj


def _group_normalize(adj):
    """
    group normalization of adjacency matrix.
    shape is (K, W, V)
    """
    mask = adj > 0
    factor = np.sqrt(np.maximum(np.sum(mask, axis=1, keepdims=True) * np.sum(mask, axis=2, keepdims=True), 1))
    assert factor.shape == adj.shape
    return adj / factor


def _joint_normalize(adj):
    """
    joint normalization of adjacency matrix.
    shape is (K, W, V)
    """
    mask = adj > 0
    mask = np.sum(mask, axis=0, keepdims=True)
    factor = np.sqrt(np.maximum(np.sum(mask, axis=1, keepdims=True) * np.sum(mask, axis=2, keepdims=True), 1))
    assert factor.shape == (1, *(adj.shape[1:]))
    return adj / factor


def _symmbreak(adj, vertices):
    """
    break the symmetry in adjacency matrix.

    input (K, W, V)
    output (2K-1, W, V)
    """
    K, W, V = adj.shape
    assert len(vertices) == W

    mask = np.ones((2 * K - 1, V, V))
    for k in range(1, K):
        m = np.triu(np.random.randn(V, V) > 0, 1) * 1.
        m += np.tril(np.ones((V, V)) - m.transpose(), -1)
        mask[k, :, :] = m
        mask[K + k - 1, :, :] = m.transpose()

    new_adj = np.concatenate((adj, adj[1:, :, :]), axis=0) * mask[:, vertices, :]
    assert (sum(adj, 0) == sum(new_adj, 0)).all()

    return new_adj


def _symmbreak_radial(adj, vertices, g):
    """
    break the symmetry in adjacency matrix from center.

    input (K, W, V)
    output (2K-1, W, V)
    """
    K, W, V = adj.shape
    assert len(vertices) == W
    root = np.argmax(g.betweenness())
    a = np.array([[len(x)-1 for x in g.get_shortest_paths(root)]])
    a = (a < a.transpose()) + 0

    mask = np.ones((2 * K - 1, V, V))
    for k in range(1, K):
        m = np.triu(a, 1) * 1.
        m += np.tril(np.ones((V, V)) - m.transpose(), -1)
        mask[k, :, :] = m
        mask[K + k - 1, :, :] = m.transpose()

    new_adj = np.concatenate((adj, adj[1:, :, :]), axis=0) * mask[:, vertices, :]
    assert (sum(adj, 0) == sum(new_adj, 0)).all()

    return new_adj
