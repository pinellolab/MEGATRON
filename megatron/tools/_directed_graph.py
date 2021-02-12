"""Functions for shortest-path-based direct graph distance"""

import numpy as np
import pandas as pd
import itertools
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
from scipy.spatial import distance

from ._utils import reorder_list


def _directed_graph(mat_clone,
                    mat_coord,
                    df_time,
                    radius=None,
                    min_cells=2,
                    eps=None
                    ):
    """Shortest-path-based directed graph

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

    df_clones = pd.DataFrame(index=df_time.index,
                             data=mat_clone.A)
    df_coord = pd.DataFrame(index=df_time.index,
                            data=mat_coord)
    if radius is None:
        radius = np.mean(df_coord.max(axis=0) - df_coord.min(axis=0))/5
        print(f'Estimated radius is {radius}')
    G, df_nodes = _build_graph(df_clones,
                               df_coord,
                               df_time,
                               radius,
                               min_cells=min_cells,
                               eps=eps)
    list_Timepoints = list(np.sort(np.unique(df_time)))
    col_coord = df_coord.columns.tolist()
    list_dist = []
    for p_clone in itertools.combinations(df_clones.columns, 2):
        nodes_i = df_nodes[df_nodes['clone'] == p_clone[0]].index.tolist()
        nodes_j = df_nodes[df_nodes['clone'] == p_clone[1]].index.tolist()
        dist_ij = _get_dist(G,
                            nodes_i,
                            nodes_j,
                            df_nodes,
                            list_Timepoints,
                            col_coord)
        list_dist.append(dist_ij)
    return list_dist


def _build_graph(df_clones,
                 df_coord,
                 df_time,
                 radius,
                 min_cells=2,
                 eps=None):
    """build graph for each clone
    Parameters
    ----------
    Returns
    -------
    """

    G = nx.DiGraph()
    if(eps is None):
        eps = radius
    list_Timepoints = list(np.sort(np.unique(df_time)))

    # add nodes
    for x in df_clones.columns.unique():
        cells_x = df_clones.index[df_clones[x] > 0]
        g_clone_x = df_time.loc[cells_x, ].groupby(
            df_time.loc[cells_x, ])
        timepoints_x = reorder_list(g_clone_x.groups.keys(), list_Timepoints)
        for t in timepoints_x:
            cells_x_t = g_clone_x.get_group(t).index.tolist()
            df_coord_xt = df_coord.loc[cells_x_t, ].values
            if(len(cells_x_t) >= min_cells):
                clust_t = DBSCAN(eps=eps,
                                 min_samples=min_cells).fit(df_coord_xt)
                if(len(clust_t.core_sample_indices_) > 0):
                    for clust_id in np.unique(
                            clust_t.labels_[clust_t.core_sample_indices_]):
                        label_mask = (clust_t.labels_ == clust_id)
                        G.add_node(
                            f'clone{x}_time_{t}_cluster_{clust_id}',
                            pos=np.median(df_coord_xt[label_mask], axis=0),
                            n_cells=sum(label_mask),
                            cells=list(np.array(cells_x_t)
                                       [clust_t.labels_ == clust_id]),
                            time=t,
                            clone=x,
                            )
                else:
                    # when all cells are classfied as outliers,
                    # we calculate their median value
                    clust_id = 0
                    label_mask = (clust_t.labels_ == clust_id)
                    G.add_node(f'clone{x}_time_{t}_cluster_{clust_id}',
                               pos=np.median(df_coord_xt, axis=0),
                               n_cells=sum(label_mask),
                               cells=cells_x_t,
                               time=t,
                               clone=x,
                               )
            else:
                # when the number of cells within one timepoint is
                # smaller than min_cells, we set min_cells = 1.
                # In this case no cells will be considered as outliers
                clust_t = DBSCAN(eps=eps, min_samples=1).fit(df_coord_xt)
                for clust_id in np.unique(
                        clust_t.labels_[clust_t.core_sample_indices_]):
                    label_mask = (clust_t.labels_ == clust_id)
                    G.add_node(
                        f'clone{x}_time_{t}_cluster_{clust_id}',
                        pos=np.median(df_coord_xt[label_mask], axis=0),
                        n_cells=sum(label_mask),
                        cells=list(np.array(cells_x_t)
                                   [clust_t.labels_ == clust_id]),
                        time=t,
                        clone=x,
                        )
    col_coord = df_coord.columns.tolist()
    df_nodes = pd.DataFrame(columns=col_coord + ['time', 'clone'])
    for x_node in G.nodes():
        df_nodes.loc[x_node] = ''
        df_nodes.loc[x_node, col_coord] = G.nodes[x_node]['pos']
        df_nodes.loc[x_node, 'time'] = G.nodes[x_node]['time']
        df_nodes.loc[x_node, 'clone'] = G.nodes[x_node]['clone']

    # add edges
    kdtree = KDTree(df_nodes[col_coord].values,
                    metric='minkowski',
                    p=2)

    # build KNN clone by clone
    for x in df_clones.columns.unique():
        nodes_x = df_nodes['clone'][df_nodes['clone'] == x].index
        g_clone_x = df_nodes.loc[nodes_x, ].groupby('time')
        timepoints_x = reorder_list(g_clone_x.groups.keys(), list_Timepoints)
        dist_nodex_x = distance.cdist(
            df_nodes.loc[nodes_x, col_coord],
            df_nodes.loc[nodes_x, col_coord],
            'minkowski',
            p=2)
        dist_nodes_x = pd.DataFrame(dist_nodex_x,
                                    index=nodes_x,
                                    columns=nodes_x)

        # build MST within the same time point
        for t in timepoints_x:
            nodes_t = g_clone_x.get_group(t).index.tolist()
            if(len(nodes_t) > 1):
                # make sure there are at least two nodes
                G_t = nx.Graph()
                G_t.add_nodes_from(nodes_t)
                G_t.add_weighted_edges_from(
                    [(pair_x[0],
                      pair_x[1],
                      dist_nodes_x.loc[pair_x[0], pair_x[1]])
                     for pair_x in itertools.combinations(nodes_t, 2)],
                    weight='dist')
                Tree_t = nx.minimum_spanning_tree(G_t, weight='dist')
                G.add_edges_from(Tree_t.to_directed().edges(data=True))

        for pair_tp in list(zip(timepoints_x[:-1], timepoints_x[1:])):
            nodes_x_y1 = g_clone_x.get_group(pair_tp[0]).index.tolist()
            nodes_x_y2 = g_clone_x.get_group(pair_tp[1]).index.tolist()
            # find the closest cell in y1 for each cell in y2
            dist_y1_y2 = dist_nodes_x.loc[nodes_x_y1, nodes_x_y2].values
            for i_y2 in range(dist_y1_y2.shape[1]):
                id_nb_y2 = np.argmin(dist_y1_y2[:, i_y2])
                G.add_edge(nodes_x_y1[id_nb_y2],
                           nodes_x_y2[i_y2],
                           dist=dist_y1_y2[id_nb_y2, i_y2],
                           color='r')

            # build KNN between two adjacent timepoints
            # find nearest neighbors within the cells from both y1 and y2
            nodes_x_y = nodes_x_y1 + nodes_x_y2
            ind_xy, dist_xy = kdtree.query_radius(
                df_nodes.loc[nodes_x_y, col_coord].values,
                r=radius,
                return_distance=True)
            for i, nxy in enumerate(nodes_x_y):
                ind_xy_i = ind_xy[i]
                dist_xy_i = dist_xy[i]
                if(ind_xy_i.shape[0] > 0):
                    # select cells within pair_tp and exclude nxy itself
                    mask_i = np.isin(df_nodes.index[ind_xy_i],
                                     list(set(nodes_x_y)-set([nxy])))
                    ind_sel = ind_xy_i[mask_i]
                    dist_sel = dist_xy_i[mask_i]
                    if(len(ind_sel) > 0):
                        for ii, xx in enumerate(ind_sel):
                            if(list_Timepoints.index(
                                    df_nodes['time'].iloc[xx]) >
                               list_Timepoints.index(
                                    df_nodes.loc[nxy, 'time'])):
                                G.add_edge(nxy,
                                           df_nodes.index[xx],
                                           dist=dist_sel[ii],
                                           color='r')
                            elif(list_Timepoints.index(
                                    df_nodes['time'].iloc[xx]) <
                                 list_Timepoints.index(
                                    df_nodes.loc[nxy, 'time'])):
                                G.add_edge(df_nodes.index[xx],
                                           nxy,
                                           dist=dist_sel[ii],
                                           color='r')
                            else:
                                G.add_edge(df_nodes.index[xx],
                                           nxy,
                                           dist=dist_sel[ii],
                                           color='green')
                                G.add_edge(nxy,
                                           df_nodes.index[xx],
                                           dist=dist_sel[ii],
                                           color='green')
    return G, df_nodes


def _order_nodes(nodes,
                 G,
                 list_Timepoints):
    list_index = []
    for x in nodes:
        list_index.append(list_Timepoints.index(G.nodes[x]['time']))
    nodes_sorted = np.array(nodes)[np.argsort(list_index)].tolist()
    return nodes_sorted


# def _find_nearest_next_timepoint(ti,
#                                  list_tj,
#                                  list_tij,
#                                  G_new):
#     # both list_tj and list_tij are sorted based on reference timepoints
#     id_i = list_tij.index(ti)
#     ti_nn = None
#     for x in list_tj:
#         if(list_tij.index(x) >= id_i):
#             ti_nn = x
#             break
#     return ti_nn


def _find_mutual_nearest_timepoints(list_ti, list_tj, list_Timepoints):

    df_mnn = pd.DataFrame(columns=['i', 'j'])  # mutual nearest neighbors
    kdtree_i = KDTree(np.reshape(
        [list_Timepoints.index(i) for i in list_ti],
        (len(list_ti), 1)))
    kdtree_j = KDTree(np.reshape(
        [list_Timepoints.index(j) for j in list_tj],
        (len(list_tj), 1)))

    dict_nn_i = dict()
    for x_i in list_ti:
        # the distance from x_i to its nearest neighbor in list_tj
        dist_nn = kdtree_j.query(
            np.reshape(list_Timepoints.index(x_i),
                       (-1, 1)),
            k=1,
            return_distance=True)[0][0, 0]
        ind_nn = kdtree_j.query_radius(
            np.reshape(list_Timepoints.index(x_i),
                       (-1, 1)),
            r=dist_nn)
        dict_nn_i[x_i] = [list_tj[j] for j in ind_nn[0]]

    dict_nn_j = dict()
    for x_j in list_tj:
        # the distance from x_j to its nearest neighbor in list_ti
        dist_nn = kdtree_i.query(
            np.reshape(list_Timepoints.index(x_j), (-1, 1)),
            k=1,
            return_distance=True)[0][0, 0]
        ind_nn = kdtree_i.query_radius(
            np.reshape(list_Timepoints.index(x_j), (-1, 1)),
            r=dist_nn)
        dict_nn_j[x_j] = [list_ti[i] for i in ind_nn[0]]

    for x_i in dict_nn_i.keys():
        for x_j in dict_nn_i[x_i]:
            if(x_i in dict_nn_j[x_j]):
                df_mnn.loc[df_mnn.shape[0], :] = [x_i, x_j]
    return df_mnn


def _get_dist(G,
              nodes_i,
              nodes_j,
              df_nodes,
              list_Timepoints,
              col_coord):
    nodes_ij = nodes_i + nodes_j
    G_new = G.subgraph(nodes_ij).copy()
    list_tij = reorder_list(
        list(df_nodes.loc[nodes_ij, 'time'].unique()), list_Timepoints)
    list_ti = reorder_list(
        list(df_nodes.loc[nodes_i, 'time'].unique()), list_Timepoints)
    list_tj = reorder_list(
        list(df_nodes.loc[nodes_j, 'time'].unique()), list_Timepoints)

    dist_nodes_ij = distance.cdist(
        df_nodes.loc[nodes_ij, col_coord],
        df_nodes.loc[nodes_ij, col_coord],
        'minkowski',
        p=2)
    dist_nodes_ij = pd.DataFrame(dist_nodes_ij,
                                 index=nodes_ij,
                                 columns=nodes_ij)

    # find mutual nearest timepoints
    df_mnn = _find_mutual_nearest_timepoints(list_ti, list_tj, list_Timepoints)

    # find mutual nearest nodes
    for x in df_mnn.index:
        ti = df_mnn.loc[x, 'i']
        tj = df_mnn.loc[x, 'j']
        nds_i = list(np.array(nodes_i)[df_nodes.loc[nodes_i, 'time'] == ti])
        nds_j = list(np.array(nodes_j)[df_nodes.loc[nodes_j, 'time'] == tj])
        dist_x = dist_nodes_ij.loc[nds_i, nds_j].values
        min_dist_x = np.min(dist_x)
        idx = np.where(dist_x == min_dist_x)
        for ii in range(idx[0].shape[0]):
            G_new.add_edge(nds_i[idx[0][ii]],
                           nds_j[idx[1][ii]],
                           dist=min_dist_x)
            G_new.add_edge(nds_j[idx[1][ii]],
                           nds_i[idx[0][ii]],
                           dist=min_dist_x)

    dict_path_length = dict(
        nx.all_pairs_dijkstra_path_length(G_new, weight='dist'))
    for x in dict_path_length.keys():
        # exclude the node itself
        del dict_path_length[x][x]

    dict_shortest_path = dict()
    for x_i in nodes_i:
        paths_to_j = [
            dict_path_length[x_i][x_j] for x_j in dict_path_length[x_i].keys()
            if x_j in nodes_j]
        if(len(paths_to_j) > 0):
            dict_shortest_path[x_i] = min(paths_to_j)
    for x_j in nodes_j:
        paths_to_i = [
            dict_path_length[x_j][x_i] for x_i in dict_path_length[x_j].keys()
            if x_i in nodes_i]
        if(len(paths_to_i) > 0):
            dict_shortest_path[x_j] = min(paths_to_i)

    # nodes that can't reach any nodes from the other clone
    nodes_ext = list(set(nodes_ij) - set(dict_shortest_path.keys()))
    if(len(nodes_ext) > 0):
        nodes_ext = _order_nodes(nodes_ext, G_new, list_Timepoints)
        for x_ext in nodes_ext:
            nb_edges = [
                (x, x_ext) for x in list(G_new.predecessors(x_ext))
                if G_new.nodes[x_ext]['time'] != G_new.nodes[x]['time']]
            if(len(nb_edges) > 0):
                id_min = np.argmin(
                    [G_new.edges[xx]['dist'] for xx in nb_edges])
                dict_shortest_path[x_ext] = \
                    dict_shortest_path[nb_edges[id_min][0]]\
                    + G_new.edges[nb_edges[id_min]]['dist']

    dict_t_len = dict()
    for t in list_tij:
        nodes_i_t = df_nodes.loc[nodes_i, ][
            df_nodes.loc[nodes_i, 'time'] == t].index.tolist()
        nodes_j_t = df_nodes.loc[nodes_j, ][
            df_nodes.loc[nodes_j, 'time'] == t].index.tolist()
        len_t = 0
        if(len(nodes_i_t) > 0):
            for x_i in nodes_i_t:
                len_t = len_t + dict_shortest_path[x_i]
        if(len(nodes_j_t) > 0):
            for x_j in nodes_j_t:
                len_t = len_t + dict_shortest_path[x_j]
        if(len_t > 0):
            dict_t_len[t] = np.float(len_t)/(len(nodes_i_t)+len(nodes_j_t))

    clone_dist = np.average(list(dict_t_len.values()))
    return clone_dist
