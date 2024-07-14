import torch
import torch.nn as nn
from torch.nn.functional import interpolate
import numpy as np
from .util import to_tuple


class GraphConvTranspose(nn.Module):
    """
    Graph Deconvolution. The dim layout are (batch, channel, graph, spatial/temporal...).
    Examples:
    * (batch, channel, graph)
    * (batch, channel, graph, time)
    * (batch, channel, graph, height, width)

    It cannot compute the output graph and do not need input the input graph.
    Instead, please specify the graph computation kernel directly,
    or specify the computation of the corresponding *Convolution* manually.

    Args:
        extra_dim (int): Extra spatial/temporal dim for graph conv, supported values are 0, 1, 2
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the graph convolution
        kernel_size (int or tuple): Graph dim supports kernel size. The first element is graphical kernel size
        stride (int or tuple): Graph dim supports stride. For the graph dim, should have kernel >= stride
                               Currently only stride 1,2 are supported
                               When stride = 1, only 'auto' and 'manual' modes are allowed
        padding (int or tuple): For the graph dim, should be 0
        output_padding (int or tuple): For the graph dim, should be 0
        dilation (int or tuple): Graph dim supports dilation
        groups (int): Graph dim supports group channels
        bias (bool): Bias
        spectral_norm (bool): Enable spectral norm
        computation_kernel (np array): Computation kernel of the corresponding *Covolution*
        upsampling (bool): Whther use upsampling for non-graph dims. Kernel should be 1, and stride is scale factor
        dynamic (bool): Default False. Not implemented yet

    Outputs:
    """

    def __init__(self,
                 extra_dim,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 spectral_norm=False,
                 computation_kernel=None,
                 symmetry_break=False,
                 upsampling=False,
                 upsampling_learn=True,
                 upsampling_mode='nearest',
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
        self.output_padding = to_tuple(output_padding, extra_dim + 1)
        self.dilation = to_tuple(dilation, extra_dim + 1)
        self.groups = groups
        self.bias = bias
        self.spectral_norm = spectral_norm
        self.computation_kernel = computation_kernel
        self.symmetry_break = symmetry_break
        self.upsampling = upsampling
        self.upsampling_learn = upsampling_learn
        self.upsampling_mode = upsampling_mode
        self.dynamic = dynamic
        self.use_graph_conv = use_graph_conv
        self.use_node_bias = use_node_bias

        # sanity check
        if dynamic: raise NotImplementedError()
        assert extra_dim in [0, 1, 2]
        assert self.padding[0] == 0
        assert self.output_padding[0] == 0
        assert self.kernel_size[0] >= self.stride[0]
        assert self.stride[0] in [1, 2]
        assert computation_kernel is not None
        # assert computation_kernel.shape[0] == (
        #     (self.kernel_size[0] - 1) * self.dilation[0] + 1) * 2**symmetry_break - symmetry_break
        if upsampling:
            assert (np.array(self.kernel_size[1:]) == np.ones_like(self.kernel_size[1:])).all()
            if not upsampling_learn:
                assert in_channels == out_channels
                self.computation_kernel = np.sum(self.computation_kernel, axis=0, keepdims=True)

        self.computation_kernel_size, self.in_vertices_num, self.out_vertices_num = self.computation_kernel.shape

        A = torch.tensor(self.computation_kernel, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        if extra_dim == 0:
            DeConv = nn.ConvTranspose1d
        elif extra_dim == 1:
            DeConv = nn.ConvTranspose2d
        elif extra_dim == 2:
            DeConv = nn.ConvTranspose3d
        else:
            raise ValueError('extra dim must in 0, 1, 2')

        if use_graph_conv:
            num_node_kernel = 1
        else:
            num_node_kernel = self.in_vertices_num

        if upsampling:
            if upsampling_learn:
                self.deconv = DeConv(
                    in_channels * num_node_kernel,
                    out_channels * self.computation_kernel_size * num_node_kernel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    output_padding=0,
                    dilation=1,
                    groups=self.groups * num_node_kernel,
                    bias=bias)
            else:
                self.deconv = None
        else:
            self.deconv = DeConv(
                in_channels * num_node_kernel,
                out_channels * self.computation_kernel_size * num_node_kernel,
                kernel_size=(1, *(self.kernel_size[1:])),
                stride=(1, *(self.stride[1:])),
                padding=(0, *(self.padding[1:])),
                output_padding=(0, *(self.output_padding[1:])),
                dilation=(1, *(self.dilation[1:])),
                groups=self.groups * num_node_kernel,
                bias=bias)

        if spectral_norm and self.deconv is not None:
            self.deconv = nn.utils.spectral_norm(self.deconv)

        if use_node_bias:
            bias_data = torch.Tensor(
                self.out_channels, self.out_vertices_num)
            bias_data.fill_(0)
            self.node_bias = nn.Parameter(bias_data)

    def forward(self, x, graph=None):
        """
        Input shape: (batch, channel, graph, spatial/temporal)
                        n       c       w       ab  /   t

        Output shape: (batch, channel, graph, spatial/temporal)
                         n       c       v       ab  /   t
        """
        if graph is not None: raise NotImplementedError()
        c, k, w = self.out_channels, self.computation_kernel_size, self.out_vertices_num
        if self.deconv is not None:
            if self.use_graph_conv:
                x = self.deconv(x)
            else:
                assert len(x.size()) == 4
                n, c_in, v, t = x.size()
                x = x.permute(0, 2, 1, 3).contiguous().view(n, v*c_in, 1, t)
                x = self.deconv(x)
                x = x.view(n, v, c*k, x.size(3)).permute(0, 2, 1, 3)
        else:
            assert k == 1

        if self.extra_dim == 0:
            n, ck, v = x.size()
            assert ck == c * k
            x = x.view(n, c, k, v)
            x = torch.einsum('kwv,nckw->ncv', (self.A.data, x)).contiguous()
            assert x.size() == (n, c, w)
        elif self.extra_dim == 1:
            n, ck, v, t = x.size()
            assert ck == c * k
            x = x.view(n, c, k, v, t)
            x = torch.einsum('kwv,nckwt->ncvt', (self.A.data, x)).contiguous()
            assert x.size() == (n, c, w, t)
            if self.upsampling:
                st, pt = self.stride[1], self.padding[1]
                x = x.view(n, c * w, t)
                x = interpolate(x, mode=self.upsampling_mode, scale_factor=st).view(n, c, w, st * t)
                x = x[:, :, :, pt:(st * t - pt)]
        elif self.extra_dim == 2:
            n, ck, v, a, b = x.size()
            assert ck == c * k
            x = x.view(n, c, k, v, a, b)
            x = torch.einsum('kwv,nckwab->ncvab', (self.A.data, x)).contiguous()
            assert x.size() == (n, c, w, a, b)
            if self.upsampling:
                (sa, sb), (pa, pb) = self.stride[1:], self.padding[1:]
                x = x.view(n, c * w, a, b)
                x = interpolate(x, mode=self.upsampling_mode, scale_factor=(sa, sb)).view(n, c, w, sa * a, sb * b)
                x = x[:, :, :, pa:(sa * a - pa), pb:(sb * b - pb)]

            x = self.upsample(x)

        if self.use_node_bias:
            assert len(x.size()) == 4
            x = x + self.node_bias.unsqueeze(0).unsqueeze(3)

        return x
