"""Functions for shortest-path-based direct graph distance"""

import numpy as np
import pandas as pd
import itertools
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform

from ._utils import sort_list


def _graph_distance(ad_input,
                    k=3,
                    metric='euclidean',
                    ):
    """Shortest-path-based graph

    Parameters
    ----------
    mat_clone: `array-like`
        Cells by clones relation matrix.
    mat_coord: `array-like`
        Cell coordinate matrix
    df_time: `pd.DataFrame`
        Temporal information of cells
    radius: `float`
        Limiting distance of neighbors to return
    min_cells: `int`
        The minimum number of cells for each node(cluster)
    eps: `float`
        eps for DBSCAN clustering
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other.

    Returns
    -------
    mat_dist: `array-like`
        A condensed clone distance matrix.
        It can be converted into a redundant square matrix using `squareform`
        from Scipy.
    """

    G = _build_graph(ad_input,
                     k=k,
                     metric=metric)


def _build_graph(ad_input,
                 k=3,
                 metric='euclidean'):
    """build graph for each clone
    Parameters
    ----------
    Returns
    -------
    """

    anno_time = ad_input.uns['params']['anno_time']

    G = nx.Graph()
    time_sorted = np.unique(ad_input.obs[anno_time])
    mat_clone = ad_input.X
    for i_clone, clone in enumerate(ad_input.var_names):
        cells_i = ad_input.obs_names[mat_clone[:, i_clone].nonzero()[0]]
        ad_input_i = ad_input[cells_i, ]
        ad_input_i.obsp['dist'] = \
            squareform(pdist(ad_input_i.obsm['X_coord'], metric=metric))
        gp_i = ad_input_i.obs.groupby(anno_time)
        time_sorted_i = sort_list(gp_i.groups.keys(), time_sorted)
        if len(time_sorted_i) > 1:
            for t1, t2 in list(zip(time_sorted_i[:-1], time_sorted_i[1:])):
                cells_i_t = gp_i.get_group(t1).index.tolist() \
                    + gp_i.get_group(t2).index.tolist()
                # knn
                k_ = min(k, len(cells_i_t))
                mat_dis_i_t = ad_input_i[cells_i_t, ].obsp['dist']
                nbrs = NearestNeighbors(n_neighbors=k_,
                                        metric='precomputed').fit(mat_dis_i_t)
                mat_knn = nbrs.kneighbors_graph(mode='distance')
                G_i_t = nx.from_scipy_sparse_matrix(mat_knn,
                                                    edge_attribute='dist')
                mapping = dict(zip(np.arange(len(cells_i_t)), cells_i_t))
                G_i_t = nx.relabel_nodes(G_i_t, mapping)
                G.add_edges_from(G_i_t.to_undirected().edges(data=True))
                # add MST to make sure the graph is connected
                G_i_t_complete = nx.from_scipy_sparse_matrix(
                    csr_matrix(mat_dis_i_t), edge_attribute='dist')
                mapping = dict(zip(np.arange(len(cells_i_t)), cells_i_t))
                G_i_t_complete = nx.relabel_nodes(G_i_t_complete, mapping)
                tree_i_t = nx.minimum_spanning_tree(G_i_t_complete,
                                                    weight='dist')
                G.add_edges_from(tree_i_t.to_undirected().edges(data=True))
        else:
            cells_i_t = gp_i.get_group(time_sorted_i[0]).index.tolist()
            k_ = min(k, len(cells_i_t))
            # knn
            mat_dis_i_t = ad_input_i[cells_i_t, ].obsp['dist']
            nbrs = NearestNeighbors(n_neighbors=k_,
                                    metric='precomputed').fit(mat_dis_i_t)
            mat_knn = nbrs.kneighbors_graph(mode='distance')
            G_i_t = nx.from_scipy_sparse_matrix(mat_knn,
                                                edge_attribute='dist')
            mapping = dict(zip(np.arange(len(cells_i_t)), cells_i_t))
            G_i_t = nx.relabel_nodes(G_i_t, mapping)
            G.add_edges_from(G_i_t.to_undirected().edges(data=True))
            # add MST to make sure the graph is connected
            G_i_t_complete = nx.from_scipy_sparse_matrix(
                csr_matrix(mat_dis_i_t), edge_attribute='dist')
            mapping = dict(zip(np.arange(len(cells_i_t)), cells_i_t))
            G_i_t_complete = nx.relabel_nodes(G_i_t_complete, mapping)
            tree_i_t = nx.minimum_spanning_tree(G_i_t_complete,
                                                weight='dist')
            G.add_edges_from(tree_i_t.to_undirected().edges(data=True))
    return G
