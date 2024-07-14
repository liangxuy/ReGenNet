import numpy as np
import pickle as pkl

from src.config import SMPL_KINTREE_PATH
from src.config import SMPLX_KINTREE_PATH


class Graph:
    """ The Graph to model the skeletons extracted by the openpose
    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).
        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D
        - smpl: Consists of 24/23 joints with without global rotation.
        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points
    """

    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 kintree_path=SMPL_KINTREE_PATH,
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.kintree_path = kintree_path
        
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,
                                                                        11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        # elif layout == 'smpl':
        #     self.num_node = 24
        #     self_link = [(i, i) for i in range(self.num_node)]
        #     kt = pkl.load(open(self.kintree_path, "rb"))
        #     neighbor_link = [(k, kt[1][i + 1]) for i, k in enumerate(kt[0][1:])]
        #     self.edge = self_link + neighbor_link
        #     self.center = 0
        elif layout == 'smpl':
            self.num_node = 24 + 1
            self_link = [(i, i) for i in range(self.num_node)]
            kt = pkl.load(open(self.kintree_path, "rb"))
            neighbor_link = [(k, kt[1][i + 1]) for i, k in enumerate(kt[0][1:])]
            neighbor_link.append((0, 24)) # link root rotational pose and root translation
            self.edge = self_link + neighbor_link
            self.center = 0
        elif layout == 'smpl_noglobal': # Not Tested
            self.num_node = 23
            self_link = [(i, i) for i in range(self.num_node)]
            kt = pkl.load(open(self.kintree_path, "rb"))
            neighbor_link = [(k, kt[1][i + 1]) for i, k in enumerate(kt[0][1:])]
            # remove the root joint
            neighbor_1base = [n for n in neighbor_link if n[0] != 0 and n[1] != 0]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 0
        elif layout == 'smplx': # https://meshcapade.wiki/SMPL#skeleton-layout
            self.num_node = 55 + 1
            self_link = [(i, i) for i in range(self.num_node)]
            kt = np.load(SMPLX_KINTREE_PATH, allow_pickle=True)['kintree_table']
            neighbor_link = [(k, kt[1][i + 1]) for i, k in enumerate(kt[0][1:])]
            neighbor_link.append((0, 55))
            # neighbor_link = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 22), (22, 23), (22, 24),
            #     (9, 13), (13, 16), (16, 18), (18, 20),
            #     (9, 14), (14, 17), (17, 19), (19, 21),
            #     (0, 1), (1, 4), (4, 7), (7, 10),
            #     (0, 2), (2, 5), (5, 8), (8, 11),
            #     (20, 52), (52, 53), (53, 54), (20, 40), (40, 41), (41, 42), (20, 43), (43, 44), (44, 45), (20, 49), (49, 50), (50, 51), (20, 46), (46, 47), (47, 48),
            #     (21, 37), (37, 38), (38, 39), (21, 25), (25, 26), (26, 27), (21, 28), (28, 29), (29, 30), (21, 34), (34, 35), (35, 36), (21, 31), (31, 32), (32, 33),
            #     (0, 55)] # link root rotational pose and root translation
            self.edge = self_link + neighbor_link
            self.center = 0
        elif layout == 'ntu-rgb+d': # limb or without global rotation
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [((i - 2)%self.num_node, (j - 2)%self.num_node) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link # root translation in the last dim
            self.center = 21 - 2
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        # elif layout=='customer settings'
        #     pass
        else:
            raise NotImplementedError("This Layout is not supported")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise NotImplementedError("This Strategy is not supported")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD
