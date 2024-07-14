import torch
import torch.nn as nn
from .graph_deconv import GraphConvTranspose
from .graph_conv import GraphConv


class SimpleGC(nn.Module):
    def __init__(self, in_channels, adj):
        super(SimpleGC, self).__init__()
        assert adj.shape[1] == adj.shape[2]
        num_node_kernel = adj.shape[-1]
        computation_kernel_size = adj.shape[0]
        self.num_node_kernel = num_node_kernel   # num of node
        self.computation_kernel_size = computation_kernel_size  # 2
        self.conv = nn.Conv2d(in_channels * num_node_kernel,
                              in_channels * computation_kernel_size * num_node_kernel,
                              kernel_size=1,
                              stride=1,
                              groups=num_node_kernel,
                              bias=False)
        A = torch.tensor(adj, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

    def forward(self, x):
        N, C, V, T = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(N, V * C, 1, T)
        x = self.conv(x)
        x = x.view(N, V, C * self.computation_kernel_size, T).permute(0, 2, 1, 3)
        x = x.view(N, C, self.computation_kernel_size, V, T)
        x = torch.einsum('kwv,nckwt->ncvt', (self.A.data, x)).contiguous()
        return x


class MTA(nn.Module):
    def __init__(self, channels, adj, part=4, temporal_size=5, bn=True):
        super(MTA, self).__init__()
        temporal_pad = temporal_size // 2
        convs = []
        bns = []
        gconvs = []
        self.part = part
        for i in range(part - 1):
            gconvs.append(SimpleGC(channels//4, adj))
            convs.append(nn.Conv1d(channels//4, channels//4, temporal_size, padding=temporal_pad, groups=channels//4))
            bns.append(nn.BatchNorm2d(channels//4) if bn else nn.GroupNorm(8, channels//4))
        weight = torch.zeros(channels//4, 1, temporal_size)
        weight[(channels//4 // 8):(channels//4 // 4), 0, 0] = 1.0  # right-shifted part
        weight[(channels//4 // 4):, 0, 1] = 1.0  # unchanged part
        weight[:(channels//4 // 8), 0, 2] = 1.0  # left-shifted part
        for i in range(part - 1):
            convs[i].weight = nn.Parameter(weight)
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.gconvs = nn.ModuleList(gconvs)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        # move relu after addition, like origin ResNet

    def forward(self, x):
        N, C, V, T = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(-1, C, T)   # NV, C, T
        x_splits = torch.split(x, C//4, 1)
        for i in range(self.part - 1):
            if i == 0:
                split = x_splits[i]
            else:
                split = split + x_splits[i]
            split = self.convs[i](split)
            split = split.view(N, V, C//4, T).permute(0, 2, 1, 3).contiguous()
            split = self.gconvs[i](split)
            split = self.relu(self.bns[i](split))
            if i == 0:
                out = split
            else:
                out = torch.cat((out, split), 1)
            split = split.permute(0, 2, 1, 3).contiguous().view(-1, C//4, T)

        last_split = x_splits[self.part - 1].view(N, V, C//4, T).permute(0, 2, 1, 3)
        out = torch.cat((out, last_split), 1)

        return out


class G_Block(nn.Module):
    def __init__(self, D_in, D_out, adj, refine_adj, mta_size, skeleton_kernel_size, spectral_norm,
                 symmetry_break, use_graph_conv, bias=False, up_mode='nearest'):
        super(G_Block, self).__init__()
        self.upsample = GraphConvTranspose(
            extra_dim=1,
            in_channels=D_in,
            out_channels=D_in,
            kernel_size=(skeleton_kernel_size, 1),
            stride=(2, 2),
            padding=(0, 0),
            bias=bias,
            spectral_norm=spectral_norm,
            computation_kernel=adj,
            upsampling=True,
            upsampling_learn=False,
            upsampling_mode=up_mode,
            symmetry_break=symmetry_break,
            use_graph_conv=use_graph_conv
        )

        self.SVC = GraphConvTranspose(
            extra_dim=1,
            in_channels=D_in,
            out_channels=D_out,
            kernel_size=(skeleton_kernel_size, 3),
            stride=(1, 1),
            padding=(0, 1),
            bias=False,
            spectral_norm=spectral_norm,
            computation_kernel=refine_adj,
            upsampling=False,
            upsampling_learn=False,
            upsampling_mode=up_mode,
            symmetry_break=symmetry_break,
            use_graph_conv=use_graph_conv
        )

        self.BN = nn.BatchNorm2d(D_out)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.mta = MTA(D_out, refine_adj, temporal_size=mta_size)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        if D_in == D_out:
            self.residual = lambda x: x
        else:
            self.residual = nn.Conv2d(D_in, D_out, kernel_size=1, stride=1)

    def forward(self, x):
        # upsample t or st (according to whether the adj is VxV or VxW)
        x = self.upsample(x)
        identity = x   # short cut
        x = self.SVC(x)
        x = self.relu1(self.BN(x))
        x = self.mta(x)

        identity = self.residual(identity)
        x += identity
        x = self.relu2(x)

        return x


class D_Block(nn.Module):
    def __init__(self, D_in, D_out, adj, mta_adj, skeleton_kernel_size,
                 svc_time_kernel_size, svc_time_stride, svc_time_pad, spectral_norm, symmetry_break,
                 use_graph_conv, mta_time_kernel_size, bias=False, pooling=False):
        super(D_Block, self).__init__()
        self.SVC = GraphConv(
            extra_dim=1,
            in_channels=D_in,
            out_channels=D_out,
            kernel_size=(skeleton_kernel_size, svc_time_kernel_size),
            stride=(2, svc_time_stride),
            padding=(0, svc_time_pad),
            bias=False,
            spectral_norm=spectral_norm,
            mode='manual',
            computation_kernel=adj,
            symmetry_break=symmetry_break,
            normalize='group',
            pooling=pooling,
            return_graph=False,
            use_graph_conv=use_graph_conv
        )

        self.GN = nn.GroupNorm(32, D_out)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.mta = MTA(D_out, mta_adj, temporal_size=mta_time_kernel_size, bn=False)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        if adj.shape[1] == adj.shape[2]:
            self.downsample = lambda x: x
        else:
            self.downsample = lambda x: torch.einsum('wv,ncvt->ncwt', (self.SVC.A.data.sum(0), x)).contiguous()
        if D_in == D_out:
            self.residual = lambda x: x
        else:
            self.residual = nn.Conv2d(D_in, D_out, kernel_size=1, stride=(1, svc_time_stride))

    def forward(self, x):
        identity = x
        x = self.SVC(x)
        x = self.relu1(self.GN(x))
        x = self.mta(x)

        identity = self.residual(self.downsample(identity))
        x += identity
        x = self.relu2(x)

        return x


if __name__ == '__main__':
    import numpy as np
    adj = np.random.rand(2, 11, 25)
    refine_adj = np.random.rand(2, 11, 11)
    # mta = MTA(64, adj)
    x = torch.rand([16, 64, 25, 64])
    # x = mta(x)
    # gc = SimpleGC(64, adj)
    # x = gc(x)
    # block = G_Block(64, 32, adj, refine_adj, 5, 2, False, 'radial',
    #                 False)
    # x = block(x)

    block1 = D_Block(32, 64, adj, refine_adj, 2, 4, 2, 1, False, 'radial', False, 5)
    x = block1(x)