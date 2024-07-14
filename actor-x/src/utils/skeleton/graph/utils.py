import igraph as ig
import numpy as np


def adjacency2graph(adj):
    """
    Convert an adjacency matrix to an undirected graph object
    """
    return ig.Graph.Adjacency((adj > 0).tolist()).as_undirected()


def edges2graph(edges):
    """
    Convert an edge list to an undirected graph object
    """
    g = ig.Graph()
    num_vertices = max([max(v) for v in edges]) + 1
    g.add_vertices(num_vertices)
    g.add_edges(edges)
    return g


def graph2adjacency(graph):
    """
    Convert an graph object to adjacency matrix
    """
    return np.array(graph.get_adjacency().data)


def _large_influence(size, ker, stride):
    """
    Large influence of deconv
    """
    assert ker >= stride
    return (size - 1) * stride + ker


def _small_influence(size, ker, stride):
    """
    Small influence of deconv
    """
    assert ker % stride == 0
    return _large_influence(size, ker, stride) - 2 * (ker - stride), ker - stride


def large_influence(net, input):
    print(f'large influence, input {input}')
    pad = 1
    for k, s in net:
        input = _large_influence(input, k, s)
        pad *= s
        assert input > 0
        print(f'ker {k}, stride {s}, pad {pad}, output {input}')
    return input, pad


def small_influence(net, input):
    print(f'small influence, input {input}')
    offset = 1
    for k, s in net:
        input, pad = _small_influence(input, k, s)
        offset *= s
        assert input > 0
        print(f'ker {k}, stride {s}, pad {pad}, offset {offset}, output {input}')
    return input
