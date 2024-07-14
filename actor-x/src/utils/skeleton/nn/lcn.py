import torch
import torch.nn as nn
import numpy as np


class LCN(nn.Module):
    def __init__(self, in_channel, out_channel, A, down_A=None, temporal_kernel=3,
                 time_stride=1, time_pad=None, bias=True):
        """

        :param in_channel:
        :param out_channel:
        :param A: the adjacency matrix(has self link) one kernel per node, the receptive field is the 1-NN
        :param unique_node_index:
        :param node_index2kernel_index:
        """
        super(LCN, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.temporal_kernel = temporal_kernel
        self.time_stride = time_stride
        if time_pad is not None:
            self.time_pad = time_pad
        else:
            self.time_pad = self.temporal_kernel // 2
        self.bias = bias

        if len(A.shape) == 3:
            A = np.sum(A, axis=0)
        self.computation_kernel = A
        A = torch.tensor(self.computation_kernel, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        self.down_A = down_A
        if down_A is not None:
            if len(down_A.shape) == 3:
                self.down_computation_kernel = np.sum(down_A, axis=0)
            else:
                self.down_computation_kernel = down_A
            A_down = torch.tensor(self.down_computation_kernel, dtype=torch.float32, requires_grad=False)
            self.register_buffer('A_down', A_down)

        self.receptive_field = self.get_receptive_field()
        self.conv_group = self.get_conv_group()

    def get_receptive_field(self):
        receptive_field = list()
        assert len(self.A.shape) == 2
        assert self.A.shape[0] == self.A.shape[1]
        num_node = self.A.shape[0]
        for i in range(num_node):
            receptive_field.append(np.where(self.A[i] != 0)[0])
        return receptive_field

    def get_conv_group(self):
        conv_group = list()
        for neigbour_node_list in self.receptive_field:
            conv_group.append(nn.Conv2d(self.in_channel, self.out_channel,
                                    kernel_size=(len(neigbour_node_list), self.temporal_kernel),
                                    padding=(0, self.time_pad),
                                    stride=(1, self.time_stride), bias=self.bias))
        return nn.ModuleList(conv_group)

    def forward(self, x):
        N, C, V, T = x.shape
        output = list()
        for i in range(V):
            neighbor_index = self.receptive_field[i]
            neighbor_feature = x[:, :, neighbor_index, :]
            output.append(self.conv_group[i](neighbor_feature))
        output = torch.cat(output, dim=2)
        if self.down_A is not None:
            output = torch.einsum('wv,ncvt->ncwt', (self.A_down.data, output))
        return output


if __name__ == '__main__':
    A = np.random.rand(25, 25)
    down_A = np.random.rand(11, 25)
    A[A < 0.5] = 0

    model = LCN(3, 64, A, deconv=True)
    # model.test()
    x = torch.rand([16, 3, 25, 64])
    output = model(x)
    print(output.shape)
    # model_path = 'IKGCN_model.pt'
    # state_dict = model.state_dict()
    # print(state_dict)
    # weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
    # torch.save(weights, model_path)
