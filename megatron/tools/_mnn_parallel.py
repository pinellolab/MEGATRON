import numpy as np
from sklearn.neighbors import radius_neighbors_graph, kneighbors_graph
from scipy.spatial.distance import squareform
import multiprocessing


def _mnn_parallel(mat_clone,
                 mat_coord,
                 df_time,
                 dist="kneighbors",
                 radius=1.0,
                 neighbors=5,
                 mode="distance",
                 n_jobs = multiprocessing.cpu_count()):
    time_steps = np.unique(df_time)
    num_clones = mat_clone.shape[1]
    num_dim = mat_coord.shape[1]
    distance_matrix = np.zeros((num_clones, num_clones))
    num_noninform = 0

    global rng
    print("Global neighbors graph variable instantiated")
    if dist=="kneighbors":
        rng = kneighbors_graph(mat_coord, neighbors, mode=mode, include_self=True)
        print("k-neighbors graph created")
    else:
        rng = radius_neighbors_graph(mat_coord, radius=radius, mode=mode, include_self=True)
        print("Radius neighbors graph created")

    nz = rng.nonzero()
    coords = np.stack((nz[0],nz[1]),axis=-1)
    # for O(1) lookup times
    rng = rng.todok()
    print("Graph converted to DOK format")

    j_params = []
    mapping = []
    mappingctr = 1
    for i in range(num_clones):
        mappingcol = []
        # get the coords for all cells in i
        cells_in_i = mat_clone[:,i].nonzero()[0]
        coords_for_i = mat_coord[cells_in_i]
        time_for_i = df_time[cells_in_i]

        ts_i_dict = {}
        for t in time_steps:
            ts_i = cells_in_i[np.where(time_for_i == t)[0]]
            ts_i_neighbors = coords[(np.isin(coords, ts_i)[:,0] == True)]
            total_all_ts_i_neighbors = rng[ts_i_neighbors[:,0],ts_i_neighbors[:,1]].sum()
            ts_i_dict[t] = [ts_i_neighbors, total_all_ts_i_neighbors]

        for j in range(i):
            dist = 0
            # get cells in clone j
            cells_in_j = mat_clone[:,j].nonzero()[0]
            coords_for_j = mat_coord[cells_in_j]
            time_for_j = df_time[cells_in_j]
            params = [time_steps, ts_i_dict, cells_in_j, coords, time_for_j, cells_in_i]
            mappingcol.append(mappingctr)
            mappingctr += 1
            j_params.append(params)
        mapping.append(mappingcol)
    max_len = np.max([len(a) for a in mapping]) + 1
    mapping = np.asarray([np.pad(a, (0, max_len - len(a)), 'constant', constant_values = 0) for a in mapping]).flatten(order='F')
    mapping = mapping[mapping!=0]
    mapping = mapping - 1
    mappingdict = dict(enumerate(mapping))
    pool = multiprocessing.Pool(processes=n_jobs)
    print("Pool of jobs created")
    results_unordered = pool.map(calc_dist, j_params)
    pool.close()
    pool.join()
    print("All jobs complete")
    results_ordered = [0]*len(results_unordered)
    for idx in mappingdict:
        results_ordered[idx] = results_unordered[int(mappingdict[idx])]
    return results_ordered

def calc_dist(params):
    time_steps, ts_i_dict, cells_in_j, coords, time_for_j, cells_in_i = params
    ts_dists = []
    for t in time_steps:
        ts_i_neighbors, total_all_ts_i_neighbors = ts_i_dict[t]
        ts_j = cells_in_j[np.where(time_for_j == t)[0]]
        ts_j_neighbors = coords[(np.isin(coords, ts_j)[:,0] == True)]
        total_all_ts_j_neighbors = rng[ts_j_neighbors[:,0],ts_j_neighbors[:,1]].sum()

        ts_i_neighbors_in_j = ts_i_neighbors[(np.isin(ts_i_neighbors, cells_in_j)[:,1] == True)]
        total_ts_i_neighbors_in_j = rng[ts_i_neighbors_in_j[:,0],ts_i_neighbors_in_j[:,1]].sum()

        ts_j_neighbors_in_i = ts_j_neighbors[(np.isin(ts_j_neighbors, cells_in_i)[:,1] == True)]
        total_ts_j_neighbors_in_i = rng[ts_j_neighbors_in_i[:,0],ts_j_neighbors_in_i[:,1]].sum()
        #print(total_all_ts_i_neighbors)
        #print(total_all_ts_j_neighbors)
        if total_all_ts_i_neighbors == 0 or total_all_ts_j_neighbors == 0:
            continue

        ts_i_frac_to_j = total_ts_i_neighbors_in_j / total_all_ts_i_neighbors
        ts_j_frac_to_i = total_ts_j_neighbors_in_i / total_all_ts_j_neighbors

        ts_i_frac = 1 / ((ts_i_frac_to_j)+1)
        ts_j_frac = 1 / ((ts_j_frac_to_i)+1)

        ts_dists.append(ts_i_frac)
        ts_dists.append(ts_j_frac)
    return np.mean(ts_dists)
