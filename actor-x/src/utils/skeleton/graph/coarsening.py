# pylint: disable=E1137,E1136,W0106
import igraph as ig
from .utils import adjacency2graph, graph2adjacency

COARSEN = ['stable', 'greedy']
COARSEN_STABLE_TYPE = ['betweenness', 'degree', 'closeness', 'eccentricity', 'smart']


def coarsen(adj, num_out=0, method='stable', type=None):
    """
    Graph coarsening. Support stable | gredy (not implemented yet) method.

    For stable method, the process is,
    1. find all maximal independent set
    2. filter by set cardinality if necessary
    3. select candidates with max sum of degrees
    4. select candidates with max mean of betweenesses
    5. select candidates with max mean of closeness
    6. select candidates with min mean of eccentricity
    7. select candidates with min count of vertices
    8. select candidates with min sum of vertex indices
    This process do not suit for large graphs (#vertices > 50)

    Input:
    adj: source graph adjacency matrix
    num_out: num of target graph vertices. 0 for auto selection
    method: coarsening strategy, stable | greedy (not implemented)
    type: extra argument for coarsening methods

    Output:
    Adjacency of target graph, selected vertices
    """
    assert method in COARSEN
    g = adjacency2graph(adj)

    if method == 'stable':
        g, v = coarsen_stable(g, num_out, type=type)
    elif method == 'greedy':
        g, v = coarsen_greedy(g, num_out, type=type)
    else:
        raise ValueError('coarsening method not supported')

    return graph2adjacency(g), v


def coarsen_stable(g, num_out=0, type=None, verbose=False):
    """
    Coarsen a graph by selecting a stable set (maximal independent set)

    Input:
    """
    if type is None: type = 'smart'
    assert type in COARSEN_STABLE_TYPE
    sets = g.maximal_independent_vertex_sets()
    verbose and print('stable set candidates', len(sets))

    if num_out > 0:
        sets = [s for s in sets if len(s) == num_out]
        verbose and print(f'stable set candidates with {num_out} elements', len(sets))
        if not sets:
            raise ValueError(f'the specified number {num_out} of output graph is invaid')

    if type in ['degree', 'smart'] and len(sets) > 1:
        scores = g.degree()
        imp = [sum(scores[i] for i in v) for v in sets]
        imp_max = max(imp)
        sets = [sets[i] for i in range(len(sets)) if imp[i] == imp_max]
        verbose and print('max degree', imp_max, 'candidates', len(sets))

    if type in ['betweenness', 'smart'] and len(sets) > 1:
        scores = g.betweenness(directed=False)
        imp = [ig.mean(scores[i] for i in v) for v in sets]
        imp_max = max(imp)
        sets = [sets[i] for i in range(len(sets)) if imp[i] == imp_max]
        verbose and print('max betweenness', imp_max, 'candidates', len(sets))

    if type in ['closeness', 'smart'] and len(sets) > 1:
        scores = g.closeness()
        imp = [ig.mean(scores[i] for i in v) for v in sets]
        imp_max = max(imp)
        sets = [sets[i] for i in range(len(sets)) if imp[i] == imp_max]
        verbose and print('max closeness', imp_max, 'candidates', len(sets))

    if type in ['eccentricity', 'smart'] and len(sets) > 1:
        scores = g.eccentricity()
        imp = [-ig.mean(scores[i] for i in v) for v in sets]
        imp_max = max(imp)
        sets = [sets[i] for i in range(len(sets)) if imp[i] == imp_max]
        verbose and print('min eccentricity', -imp_max, 'candidates', len(sets))

    if type in ['smart'] and len(sets) > 1:
        imp = [-len(v) for v in sets]
        imp_max = max(imp)
        sets = [sets[i] for i in range(len(sets)) if imp[i] == imp_max]
        verbose and print('min vertex num', -imp_max, 'candidates', len(sets))

    if len(sets) > 1:
        imp = [-sum(v) for v in sets]
        imp_max = max(imp)
        sets = [sets[i] for i in range(len(sets)) if imp[i] == imp_max]
        verbose and print('min index sum', -imp_max, 'candidates', len(sets))

    vertices = sorted(sets[0])

    verbose and print(f'vertex {g.vcount()} => {len(vertices)}')

    return _coarsened_graph(g, vertices), vertices


def coarsen_greedy(g, num_out=0, type=None):
    raise NotImplementedError()


def _coarsened_graph(g, vertices):
    gg = ig.Graph()
    gg.add_vertices(len(vertices))

    for i, vi in enumerate(vertices):
        for j in range(i + 1, len(vertices)):
            p = g.get_shortest_paths(vi, vertices[j])[0]
            if len(p) <= 4:
                gg.add_edge(i, j)

    gg = gg.as_undirected()
    assert gg.is_connected()
    return gg


def coarsened_graph(adj, vertices):
    g = _coarsened_graph(adjacency2graph(adj), vertices)
    return graph2adjacency(g)
