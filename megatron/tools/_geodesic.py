"""Functions for graph-based geodesic distance"""

import numpy as np
import itertools
import collections
import networkx as nx
import multiprocessing
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, cdist, squareform

# from ._utils import sort_list


def _average_geodesic(ad_input,
                      n_clusters=80,
                      clustering='kmeans',
                      k=3,
                      metric='euclidean',
                      n_jobs=1,
                      use_weight=False,
                      weight_time=None,
                      ):
    """average geodesic distances between each pair of clones

    Parameters
    ----------
    mat_clone: `array-like`
        Cells by clones relation matrix.
    mat_coord: `array-like`
        Cell coordinate matrix

    Returns
    -------
    list_dist: `array-like`
        A condensed clone distance matrix.
        It can be converted into a redundant square matrix using `squareform`
        from Scipy.
    """

    if use_weight:
        if weight_time is not None:
            assert isinstance(weight_time, dict), "`weight_time` must be dict"
            anno_time = ad_input.uns['params']['anno_time']
            if not (set(ad_input.obs[anno_time]) == set(weight_time.keys())):
                raise ValueError("keys in `weight_time` "
                                 "do not match time annotation")

    G, mat_clust_clone = _build_graph(ad_input,
                                      n_clusters=n_clusters,
                                      clustering=clustering,
                                      k=k,
                                      metric=metric)
    list_dist = _pairwise_geodesic_dist(ad_input,
                                        G,
                                        mat_clust_clone,
                                        n_jobs=n_jobs,
                                        use_weight=use_weight,
                                        weight_time=weight_time)
    return list_dist


def _build_graph(ad_input,
                 n_clusters=100,
                 clustering='kmeans',
                 k=3,
                 metric='euclidean'):
    """build graph for each clone
    Parameters
    ----------
    Returns
    -------
    """
    anno_time = ad_input.uns['params']['anno_time']
    df_time = ad_input.obs[anno_time]
    mat_clone = ad_input.X
    mat_coord = ad_input.obsm['X_coord']

    # clustering cells
    if clustering == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(mat_coord)
        clust = kmeans.labels_
        clust_pos = kmeans.cluster_centers_
    else:
        raise ValueError(
            f'"{clustering}" is not supported yet')
    ad_input.obs['cluster'] = clust
    ad_input.uns['cluster_pos'] = clust_pos
    # build a connected graph of clusters
    G = nx.Graph()
    k_ = min(k, n_clusters)

    # Build a KNN graph
    mat_dist = squareform(pdist(clust_pos, metric=metric))
    nbrs = NearestNeighbors(n_neighbors=k_,
                            metric='precomputed').fit(mat_dist)
    mat_knn = nbrs.kneighbors_graph(mat_dist, mode='distance')
    G_knn = nx.from_scipy_sparse_matrix(mat_knn,
                                        edge_attribute='dist')
    G.add_edges_from(G_knn.to_undirected().edges(data=True))
    # add a MST to make sure the graph is connected
    G_complete = nx.from_scipy_sparse_matrix(
        csr_matrix(mat_dist), edge_attribute='dist')
    G_mst = nx.minimum_spanning_tree(G_complete,
                                     weight='dist')
    G.add_edges_from(G_mst.to_undirected().edges(data=True))

    # construct a cluster-by-clone matrix
    # that stores the converted temporal info

    # convert sorted time to integer values starting from 1
    time_sorted = np.unique(df_time)
    dict_time = {x: i+1 for i, x in enumerate(time_sorted)}
    mat_time = np.array([dict_time[x] for x in df_time.values])

    mat_clust_clone = np.zeros(shape=(n_clusters, mat_clone.shape[1]))
    for i in range(mat_clone.shape[1]):
        ind_i = mat_clone[:, i].nonzero()[0]  # indices of cells
        clust_i = clust[ind_i]
        time_i = mat_time[ind_i]
        dict_clust_i = {}
        for ii, ci in enumerate(clust_i):
            if ci not in dict_clust_i.keys():
                dict_clust_i[ci] = [time_i[ii]]
            else:
                dict_clust_i[ci].append(time_i[ii])
        for x in dict_clust_i.keys():
            mat_clust_clone[x, i] = max(set(dict_clust_i[x]),
                                        key=dict_clust_i[x].count)
    return G, mat_clust_clone


# def _build_graph(ad_input,
#                  k=3,
#                  metric='euclidean'):
#     """build graph for each clone
#     Parameters
#     ----------
#     Returns
#     -------
#     """

#     anno_time = ad_input.uns['params']['anno_time']

#     G = nx.Graph()
#     time_sorted = np.unique(ad_input.obs[anno_time])
#     mat_clone = ad_input.X
#     # for each clone, build a graph of each timepoint
#     for i_clone, clone in enumerate(ad_input.var_names):
#         cells_i = ad_input.obs_names[mat_clone[:, i_clone].nonzero()[0]]
#         ad_input_i = ad_input[cells_i, ]
#         ad_input_i.obsp['dist'] = \
#             squareform(pdist(ad_input_i.obsm['X_coord'], metric=metric))
#         gp_i = ad_input_i.obs.groupby(anno_time)
#         time_sorted_i = sort_list(gp_i.groups.keys(), time_sorted)

#         # if there are multiple timepoints
#         if len(time_sorted_i) > 1:
#             # for each pair of adjacent timepoints
#             for t1, t2 in list(zip(time_sorted_i[:-1], time_sorted_i[1:])):
#                 cells_i_t = gp_i.get_group(t1).index.tolist() \
#                     + gp_i.get_group(t2).index.tolist()
#                 # Build a KNN graph
#                 k_ = min(k, len(cells_i_t))
#                 mat_dis_i_t = ad_input_i[cells_i_t, ].obsp['dist']
#                 nbrs = NearestNeighbors(n_neighbors=k_,
#                                         metric='precomputed').fit(mat_dis_i_t)
#                 mat_knn = nbrs.kneighbors_graph(mat_dis_i_t, mode='distance')
#                 G_i_t = nx.from_scipy_sparse_matrix(mat_knn,
#                                                     edge_attribute='dist')
#                 mapping = dict(zip(np.arange(len(cells_i_t)), cells_i_t))
#                 G_i_t = nx.relabel_nodes(G_i_t, mapping)
#                 G.add_edges_from(G_i_t.to_undirected().edges(data=True))
#                 # add a MST to make sure the graph of two adjacent graphs
#                 # is connected
#                 G_i_t_complete = nx.from_scipy_sparse_matrix(
#                     csr_matrix(mat_dis_i_t), edge_attribute='dist')
#                 mapping = dict(zip(np.arange(len(cells_i_t)), cells_i_t))
#                 G_i_t_complete = nx.relabel_nodes(G_i_t_complete, mapping)
#                 tree_i_t = nx.minimum_spanning_tree(G_i_t_complete,
#                                                     weight='dist')
#                 G.add_edges_from(tree_i_t.to_undirected().edges(data=True))

#         # if there is only one timepoint
#         else:
#             cells_i_t = gp_i.get_group(time_sorted_i[0]).index.tolist()
#             k_ = min(k, len(cells_i_t))
#             # Build a KNN graph
#             mat_dis_i_t = ad_input_i[cells_i_t, ].obsp['dist']
#             nbrs = NearestNeighbors(n_neighbors=k_,
#                                     metric='precomputed').fit(mat_dis_i_t)
#             mat_knn = nbrs.kneighbors_graph(mat_dis_i_t, mode='distance')
#             G_i_t = nx.from_scipy_sparse_matrix(mat_knn,
#                                                 edge_attribute='dist')
#             mapping = dict(zip(np.arange(len(cells_i_t)), cells_i_t))
#             G_i_t = nx.relabel_nodes(G_i_t, mapping)
#             G.add_edges_from(G_i_t.to_undirected().edges(data=True))
#             # add a MST to make sure the graph is connected
#             G_i_t_complete = nx.from_scipy_sparse_matrix(
#                 csr_matrix(mat_dis_i_t), edge_attribute='dist')
#             mapping = dict(zip(np.arange(len(cells_i_t)), cells_i_t))
#             G_i_t_complete = nx.relabel_nodes(G_i_t_complete, mapping)
#             tree_i_t = nx.minimum_spanning_tree(G_i_t_complete,
#                                                 weight='dist')
#             G.add_edges_from(tree_i_t.to_undirected().edges(data=True))
#     return G


def _cal_geodesic_dist(G,
                       mat_clust_clone,
                       use_weight,
                       dict_time_w_conv,
                       i,
                       j):
    ind_i = np.where(mat_clust_clone[:, i] > 0)[0]
    ind_j = np.where(mat_clust_clone[:, j] > 0)[0]
    mat_time_i = mat_clust_clone[ind_i, i]
    mat_time_j = mat_clust_clone[ind_j, j]
    mat_time_w_i = [dict_time_w_conv[x] for x in mat_time_i]
    mat_time_w_j = [dict_time_w_conv[x] for x in mat_time_j]

    # find mutual-nearest-time cells from two clones
    mat_time_dist_ij = cdist(mat_time_i.reshape(-1, 1),
                             mat_time_j.reshape(-1, 1),
                             metric='cityblock')
    min_time_dist = mat_time_dist_ij.min()
    mat_time_nn = np.where(mat_time_dist_ij == min_time_dist, 1, 0)

    # calculate average shortest paths for each pair of MNT nodes
    ids_ii, ids_jj = np.where(mat_time_nn == 1)
    mat_time_nn_len = np.zeros(mat_time_nn.shape)
    mat_time_nn_len.fill(np.nan)

    # unique combinations of nearest timepoints
    dict_freq_ij = \
        collections.Counter(list(zip(mat_time_i[ids_ii], mat_time_j[ids_jj])))
    for ii, jj in zip(ids_ii, ids_jj):
        mat_time_nn_len[ii, jj] = nx.shortest_path_length(G,
                                                          source=ind_i[ii],
                                                          target=ind_j[jj],
                                                          weight='dist')
        # remove the confounding factor #pairs_of_MNT
        mat_time_nn_len[ii, jj] *= \
            1/dict_freq_ij[(mat_time_i[ii], mat_time_j[jj])]
        if use_weight:
            mat_time_nn_len[ii, jj] *= max(mat_time_w_i[ii],
                                           mat_time_w_j[jj])
    # remove the confounding factor #timepoints
    dist = np.sum(mat_time_nn_len[ids_ii, ids_jj])/len(dict_freq_ij)
    return dist


# def _cal_geodesic_dist(ad_input,
#                        G,
#                        k,
#                        metric,
#                        mat_clone,
#                        mat_coord,
#                        mat_time,
#                        use_weight,
#                        mat_time_w,
#                        i,
#                        j):
#     ind_i = mat_clone[:, i].nonzero()[0]
#     ind_j = mat_clone[:, j].nonzero()[0]
#     mat_coord_i = mat_coord[ind_i, ]
#     mat_coord_j = mat_coord[ind_j, ]
#     mat_time_i = mat_time[ind_i]
#     mat_time_j = mat_time[ind_j]
#     mat_time_w_i = mat_time_w[ind_i]
#     mat_time_w_j = mat_time_w[ind_j]

#     # find mutual-knearest-neighbor cells
#     # when considering k nearest neighbors
#     # from two clones
#     k_ = min(k, len(ind_i), len(ind_j))
#     mat_dist_ij = cdist(mat_coord_i, mat_coord_j, metric=metric)
#     mat_knn_i = np.zeros(shape=mat_dist_ij.shape)
#     mat_knn_j = np.zeros(shape=mat_dist_ij.shape)
#     mat_knn_i[np.tile(
#               np.arange(mat_dist_ij.shape[0]).reshape(-1, 1), (1, k_)),
#               np.argsort(mat_dist_ij, axis=1)[:, :k_]] = 1
#     mat_knn_j[np.argsort(mat_dist_ij, axis=0)[:k_, :],
#               np.tile(np.arange(mat_dist_ij.shape[1]), (k_, 1))] = 1
#     mat_mnn = mat_knn_i + mat_knn_j
#     max_mnn = mat_mnn.max()
#     mat_mnn = np.where(mat_mnn == max_mnn, 1, 0)

#     # find mutual-nearest-time cells from two clones
#     mat_time_dist_ij = cdist(mat_time_i.reshape(-1, 1),
#                              mat_time_j.reshape(-1, 1),
#                              metric='cityblock')
#     min_time_dist = mat_time_dist_ij.min()
#     mat_time_nn = np.where(mat_time_dist_ij == min_time_dist, 1, 0)

#     # find edges that satisfy both MNN and MNT
#     mat_sum = mat_mnn + mat_time_nn
#     max_sum = mat_sum.max()
#     ids_i, ids_j = np.where(mat_sum == max_sum)

#     # connect graph i and graph j through the edges
#     # that satisfy both mnn and mnt
#     cells_i = ad_input.obs_names[ind_i]
#     cells_j = ad_input.obs_names[ind_j]
#     G_ij = G.subgraph(cells_i.tolist() + cells_j.tolist()).copy()
#     G_ij.add_weighted_edges_from(zip(cells_i[ids_i],
#                                      cells_j[ids_j],
#                                      mat_dist_ij[ids_i, ids_j]),
#                                  weight='dist')

#     # calculate average shortest paths for each pair of MNT nodes
#     ids_ii, ids_jj = np.where(mat_time_nn == 1)
#     mat_time_nn_len = np.zeros(mat_time_nn.shape)
#     mat_time_nn_len.fill(np.nan)

#     # unique combinations of nearest timepoints
#     # compute the average distance of each mutual nearest timepoint
#     dict_freq_ij = \
#         collections.Counter(mat_time_i[ids_ii] + mat_time_j[ids_jj])
#     for ii, jj in zip(ids_ii, ids_jj):
#         mat_time_nn_len[ii, jj] = nx.shortest_path_length(G_ij,
#                                                           source=cells_i[ii],
#                                                           target=cells_j[jj],
#                                                           weight='dist')
#         # remove the confounding factor #cells
#         mat_time_nn_len[ii, jj] *= \
#             1/dict_freq_ij[mat_time_i[ii] + mat_time_j[jj]]
#         if use_weight:
#             mat_time_nn_len[ii, jj] *= max(mat_time_w_i[ii],
#                                            mat_time_w_j[jj])
#     # remove the confounding factor #timepoints
#     dist = np.sum(mat_time_nn_len[ids_ii, ids_jj])/len(dict_freq_ij)
#     return dist


def _pairwise_geodesic_dist(ad_input,
                            G,
                            mat_clust_clone,
                            use_weight=False,
                            weight_time=None,
                            n_jobs=1):
    """calculate geodesic distance between each pair of clones
    Parameters
    ----------
    Returns
    -------
    """

    anno_time = ad_input.uns['params']['anno_time']
    df_time = ad_input.obs[anno_time]

    time_sorted = np.unique(ad_input.obs[anno_time])
    dict_time = {x: i+1 for i, x in enumerate(time_sorted)}
    if use_weight:
        if weight_time is None:
            print(f"`weight_time` is not speficied. The {anno_time} are "
                  f"auto-weighted based on #cells in each {anno_time}.")
            dict_n_cells = collections.Counter(df_time)
            n_cells = df_time.shape[0]
            dict_time_w = {x: dict_n_cells[x]/n_cells
                           for x in time_sorted}
        else:
            dict_time_w = weight_time.copy()
    else:
        dict_time_w = {x: 1 for x in time_sorted}
    print(f"The weights of {anno_time} are {dict_time_w}")
    # construct a dictionary of the converted time points
    dict_time_w_conv = {dict_time[x]: dict_time_w[x] for x in dict_time.keys()}
    list_ij = list(itertools.combinations(np.arange(ad_input.shape[1]), 2))
    list_param = [(G, mat_clust_clone,
                   use_weight, dict_time_w_conv,
                   i, j) for i, j in list_ij]
    with multiprocessing.Pool(processes=n_jobs) as pool:
        list_dist = pool.starmap(_cal_geodesic_dist, list_param)
    return list_dist


# def _pairwise_geodesic_dist(ad_input,
#                             G,
#                             metric='euclidean',
#                             k=3,
#                             use_weight=False,
#                             weight_time=None,
#                             n_jobs=1):
#     """calculate geodesic distance between each pair of clones
#     Parameters
#     ----------
#     Returns
#     -------
#     """

#     anno_time = ad_input.uns['params']['anno_time']
#     df_time = ad_input.obs[anno_time]
#     mat_clone = ad_input.X
#     mat_coord = ad_input.obsm['X_coord']

#     time_sorted = np.unique(ad_input.obs[anno_time])
#     dict_time = {x: i for i, x in enumerate(time_sorted)}
#     mat_time = np.array([dict_time[x] for x in df_time.values])
#     if use_weight:
#         if weight_time is None:
#             print(f"`weight_time` is not speficied. The {anno_time} are "
#                   f"auto-weighted based on #cells in each {anno_time}.")
#             dict_n_cells = collections.Counter(df_time)
#             n_cells = df_time.shape[0]
#             dict_time_w = {x: dict_n_cells[x]/n_cells
#                            for x in time_sorted}
#             print(f"The weights of {anno_time} are {dict_time_w}")
#             mat_time_w = np.array([dict_time_w[x] for x in df_time.values])
#         else:
#             mat_time_w = np.array([weight_time[x] for x in df_time.values])
#     else:
#         mat_time_w = np.ones(mat_time.shape)
#     list_ij = list(itertools.combinations(np.arange(ad_input.shape[1]), 2))
#     list_param = [(ad_input, G, k, metric, mat_clone, mat_coord,
#                    mat_time, use_weight, mat_time_w,
#                    i, j) for i, j in list_ij]
#     with multiprocessing.Pool(processes=n_jobs) as pool:
#         list_dist = pool.starmap(_cal_geodesic_dist, list_param)
#     return list_dist
