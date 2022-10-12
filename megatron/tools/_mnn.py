import numpy as np
from sklearn.neighbors import radius_neighbors_graph, kneighbors_graph
# from scipy.spatial.distance import squareform
import multiprocessing

import warnings
import pandas as pd
import sys


def _mnn(mat_clone,
         mat_coord,
         df_time,
         dist="kneighbors",
         radius=1.0,
         neighbors=30,
         mode="distance",
         n_jobs=multiprocessing.cpu_count()):

    print("Using %s CPUs" % n_jobs)
    if 'unknown' not in df_time:
        time_steps = np.unique(df_time)
    num_clones = mat_clone.shape[1]

    if dist == "kneighbors":
        rng = kneighbors_graph(mat_coord,
                               neighbors,
                               mode=mode,
                               include_self=True)
        print("k-neighbors graph created")
    else:
        rng = radius_neighbors_graph(
            mat_coord, radius=radius, mode=mode, include_self=True
        )
        print("Radius neighbors graph created")

    nz = rng.nonzero()
    coords = np.stack((nz[0], nz[1]), axis=-1)
    # for O(1) lookup times
    rng = rng.todok()
    print("Graph converted to DOK format")

    j_params = []
    mapping = []
    mappingctr = 1
    for i in range(num_clones):
        mappingcol = []

        # get the coords for all cells in clone i
        cells_in_i = mat_clone[:, i].nonzero()[0]

        # if I use annotation time
        if 'unknown' not in df_time:
            # get time vector for cells in clone i
            time_for_i = df_time[cells_in_i]
            ts_i_dict = {}
            # go through time steps
            for t in time_steps:
                # find cells at time 't'
                ts_i = cells_in_i[np.where(time_for_i == t)[0]]
                ts_i_neighbors = coords[np.isin(coords, ts_i)[:, 0]]
                total_all_ts_i_neighbors = rng[
                    ts_i_neighbors[:, 0], ts_i_neighbors[:, 1]
                ].sum()
                ts_i_dict[t] = [ts_i_neighbors, total_all_ts_i_neighbors]
        else:
            i_neighbors = coords[np.isin(coords, cells_in_i)[:, 0]]
            total_i_neighbors = rng[
                i_neighbors[:, 0], i_neighbors[:, 1]
            ].sum()


        for j in range(i):
            dist = 0
            # get cells in clone j
            cells_in_j = mat_clone[:, j].nonzero()[0]
            if 'unknown' not in df_time:
                time_for_j = df_time[cells_in_j]
                params = [
                    time_steps,
                    ts_i_dict,
                    cells_in_j,
                    coords,
                    time_for_j,
                    cells_in_i,
                    rng,
                ]
                mappingcol.append(mappingctr)
                mappingctr += 1
                j_params.append(params)
            else:
                params = [
                    i_neighbors,
                    total_i_neighbors,
                    cells_in_j,
                    coords,
                    cells_in_i,
                    rng,
                ]
                mappingcol.append(mappingctr)
                mappingctr += 1
                j_params.append(params)
        mapping.append(mappingcol)
    max_len = np.max([len(a) for a in mapping]) + 1
    mapping = np.asarray(
        [
            np.pad(a, (0, max_len - len(a)), "constant", constant_values=0)
            for a in mapping
        ]
    ).flatten(order="F")
    mapping = mapping[mapping != 0]
    mapping = mapping - 1
    mappingdict = dict(enumerate(mapping))
    pool = multiprocessing.Pool(processes=n_jobs)
    print("Pool of jobs created")
    try:
        # if not df_time.bool():
        if 'unknown' not in df_time:
        # if True:
            results_unordered = pool.map(calc_dist_time, j_params)
        else:
            results_unordered = pool.map(calc_dist_notime, j_params)
    except Exception:
        sys.exit(2)
    pool.close()
    pool.join()
    print("All jobs complete")
    results_ordered = [0] * len(results_unordered)
    for idx in mappingdict:
        results_ordered[idx] = results_unordered[int(mappingdict[idx])]
    return results_ordered


def calc_dist_time(params):
    time_steps, ts_i_dict, cells_in_j, \
        coords, time_for_j, cells_in_i, rng = params
    ts_dists = []
    g = 0
    for t in time_steps:
        ts_i_neighbors, total_all_ts_i_neighbors = ts_i_dict[t]
        ts_j = cells_in_j[np.where(time_for_j == t)[0]]
        ts_j_neighbors = coords[np.isin(coords, ts_j)[:, 0]]
        total_all_ts_j_neighbors = rng[
            ts_j_neighbors[:, 0], ts_j_neighbors[:, 1]
        ].sum()

        ts_i_neighbors_in_j = ts_i_neighbors[
            np.isin(ts_i_neighbors, cells_in_j)[:, 1]]
        total_ts_i_neighbors_in_j = rng[
            ts_i_neighbors_in_j[:, 0], ts_i_neighbors_in_j[:, 1]
        ].sum()

        ts_j_neighbors_in_i = ts_j_neighbors[
            np.isin(ts_j_neighbors, cells_in_i)[:, 1]]
        total_ts_j_neighbors_in_i = rng[
            ts_j_neighbors_in_i[:, 0], ts_j_neighbors_in_i[:, 1]
        ].sum()

        if total_all_ts_i_neighbors == 0 or total_all_ts_j_neighbors == 0:
            if total_all_ts_i_neighbors == 0:
                print("NO CELLS IN CLONE I AT TIME %s" % (t))

            if total_all_ts_j_neighbors == 0:
                print("NO CELLS IN CLONE J AT TIME %s" % (t))
            g += 1
            continue

        ts_i_frac_to_j = total_ts_i_neighbors_in_j / total_all_ts_i_neighbors
        ts_j_frac_to_i = total_ts_j_neighbors_in_i / total_all_ts_j_neighbors

        ts_i_frac = 1 / ((ts_i_frac_to_j) + 1)
        ts_j_frac = 1 / ((ts_j_frac_to_i) + 1)

        ts_dists.append(ts_i_frac)
        ts_dists.append(ts_j_frac)
    with warnings.catch_warnings():
        try:
            ts_mean = np.mean(ts_dists)
        except Warning:
            print("EEEEEEE RAISED")
            print(ts_dists)
            print(g)
            print(time_steps)
            raise Exception
    return ts_mean

def calc_dist_notime(params):
    # print("here1")
    i_neighbors, total_i_neighbors, cells_in_j, coords, cells_in_i, rng = params
    dists = []
    g = 0
    j_neighbors = coords[np.isin(coords, cells_in_j)[:, 0]]
    # print("here1.2")
    total_j_neighbors = rng[
        j_neighbors[:, 0], j_neighbors[:, 1]
    ].sum()
    # print("here1.4")

    i_neighbors_in_j = i_neighbors[
        np.isin(i_neighbors, cells_in_j)[:, 1]]
    total_i_neighbors_in_j = rng[
        i_neighbors_in_j[:, 0], i_neighbors_in_j[:, 1]
    ].sum()
    # print("here2")

    j_neighbors_in_i = j_neighbors[
        np.isin(j_neighbors, cells_in_i)[:, 1]]
    total_j_neighbors_in_i = rng[
        j_neighbors_in_i[:, 0], j_neighbors_in_i[:, 1]
    ].sum()
    # print("here2.5")
    # print(i_neighbors)
    # print(j_neighbors)
    # if i_neighbors == 0 or j_neighbors == 0:
        # if i_neighbors == 0:
            # print("NO NEIGHBORS TO CLONE I")

        # if j_neighbors == 0:
            # print("NO NEIGHBORS TO CLONE J")
        # g += 1
    # print("here3")

    i_frac_to_j = total_i_neighbors_in_j / total_i_neighbors
    j_frac_to_i = total_j_neighbors_in_i / total_j_neighbors
    # print(i_frac_to_j)

    # print("here4")
    i_frac = 1 / ((i_frac_to_j) + 1)
    j_frac = 1 / ((j_frac_to_i) + 1)
    # print(i_frac)
    # print(j_frac)
    # print("here5")

    dists.append(i_frac)
    dists.append(j_frac)
    # print(dists)
    with warnings.catch_warnings():
        try:
            mean = np.mean(dists)
        except Warning:
            print("EEEEEEE RAISED")
            print(dists)
            raise Exception
    return mean

def generate_centroids(num_centers, time_steps, mat_clone, mat_coord, df_time):
    centroid_dic = {}
    time_idx = pd.Index(df_time)
    for t in time_steps:
        centroid_dic[t] = "NaN"
        cells_at_t = time_idx.get_loc(t)
        # print(cells_at_t)
        # print(np.where(df_time==t))
        # print(mat_coord)
        # cells_at_t = df_time.loc[df_time['Day'] == t].index
        # for c in num_centers:
        t_mean = np.mean(mat_coord[cells_at_t], axis=0)
        centroid_dic[t] = t_mean
    return centroid_dic
