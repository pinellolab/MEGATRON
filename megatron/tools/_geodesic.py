"""Functions for shortest-path-based direct graph distance"""

import numpy as np
import itertools
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, cdist, squareform

from ._utils import sort_list


def _average_geodesic(ad_input,
                      k=3,
                      metric='euclidean',
                      ):
    """average geodesic distances between each pair of clones

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
    list_dist = _average_geodesic(ad_input,
                                  G,
                                  k=k,
                                  metric=metric)
    return list_dist


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


def _pairwise_geodesic_dist(ad_input,
                            G,
                            metric='euclidean',
                            k=3):
    """calculate geodesic distance between each pair of clones
    Parameters
    ----------
    Returns
    -------
    """

    anno_time = ad_input.uns['params']['anno_time']
    df_time = ad_input.obs[anno_time]
    mat_clone = ad_input.X
    mat_coord = ad_input.obsm['X_coord']

    time_sorted = np.unique(ad_input.obs[anno_time])
    dict_time = {x: i for i, x in enumerate(time_sorted)}
    mat_time = np.array([dict_time[x] for x in df_time.values])

    list_dist = []

    for i, j in list(itertools.combinations(np.arange(ad_input.shape[1]), 2)):
        ind_i = mat_clone[:, i].nonzero()[0]
        ind_j = mat_clone[:, j].nonzero()[0]
        mat_coord_i = mat_coord[ind_i, ]
        mat_coord_j = mat_coord[ind_j, ]
        mat_time_i = mat_time[ind_i]
        mat_time_j = mat_time[ind_j]

        # mutual nearest neighbors
        k_ = min(k, len(ind_i), len(ind_j))
        mat_dist_ij = cdist(mat_coord_i,
                            mat_coord_j,
                            metric=metric)
        mat_knn_i = np.zeros(shape=mat_dist_ij.shape)
        mat_knn_j = np.zeros(shape=mat_dist_ij.shape)
        mat_knn_i[np.tile(np.arange(mat_dist_ij.shape[0]).reshape(-1, 1),
                          (1, k_)),
                  np.argsort(mat_dist_ij, axis=1)[:, :k_]] = 1
        mat_knn_j[np.argsort(mat_dist_ij, axis=0)[:k_, :],
                  np.tile(np.arange(mat_dist_ij.shape[1]),
                          (k_, 1))] = 1

        # mutual nearest times
        mat_time_dist_ij = cdist(mat_time_i.reshape(-1, 1),
                                 mat_time_j.reshape(-1, 1),
                                 metric='cityblock')
        min_time_dist = mat_time_dist_ij.min()
        mat_time_nn = np.where(mat_time_dist_ij == min_time_dist, 1, 0)

        # keep edges that satisfy both mnn and mnt
        mat_sum = mat_knn_i + mat_knn_j + mat_time_nn
        max_sum = mat_sum.max()
        ids_i, ids_j = np.where(mat_sum == max_sum)

        # connect graph i and graph j
        cells_i = ad_input.obs_names[ind_i]
        cells_j = ad_input.obs_names[ind_j]
        G_ij = G.subgraph(cells_i.tolist() + cells_j.tolist()).copy()
        G_ij.add_weighted_edges_from(
            zip(cells_i[ids_i], cells_j[ids_j], mat_dist_ij[ids_i, ids_j]),
            weight='dist')

        # calculate average shortest paths for mutual-nearest-times nodes
        ids_ii, ids_jj = np.where(mat_time_nn == 1)
        mat_time_nn_len = np.zeros(mat_time_nn.shape)
        mat_time_nn_len.fill(np.nan)
        for ii, jj in zip(ids_ii, ids_jj):
            mat_time_nn_len[ii, jj] = \
                nx.shortest_path_length(G_ij,
                                        source=cells_i[ii],
                                        target=cells_j[jj],
                                        weight='dist')
        list_dist.append(np.mean(
            [np.nanmin(mat_time_nn_len, axis=1).mean(),
             np.nanmin(mat_time_nn_len, axis=0).mean()]))
