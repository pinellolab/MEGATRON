import numpy as np
from scipy.stats import wasserstein_distance, energy_distance
import multiprocessing
import sys


def _wasserstein(
    mat_clone,
    mat_coord,
    df_time,
    n_jobs=multiprocessing.cpu_count(),
    choice="wasserstein",
):
    time_steps = np.unique(df_time)
    num_clones = mat_clone.shape[1]
    num_dim = mat_coord.shape[1]
    j_params = []
    mapping = []
    mappingctr = 1
    for i in range(num_clones):
        mappingcol = []
        cells_in_i = mat_clone[:, i].nonzero()[0]
        coords_for_i = mat_coord[cells_in_i]
        time_for_i = df_time[cells_in_i]
        for j in range(i):
            cells_in_j = mat_clone[:, j].nonzero()[0]
            coords_for_j = mat_coord[cells_in_j]
            time_for_j = df_time[cells_in_j]
            params = [
                cells_in_i,
                coords_for_i,
                time_for_i,
                cells_in_j,
                coords_for_j,
                time_for_j,
                time_steps,
                num_dim,
                choice,
            ]
            mappingcol.append(mappingctr)
            mappingctr += 1
            j_params.append(params)
        mapping.append(mappingcol)
    # print(j_params)
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
    results_unordered = pool.map(calc_dist, j_params)
    pool.close()
    pool.join()
    results_ordered = [0] * len(results_unordered)
    for idx in mappingdict:
        results_ordered[idx] = results_unordered[int(mappingdict[idx])]
    return results_ordered


def calc_dist(params):
    dist = 0
    (
        cells_in_i,
        coords_for_i,
        time_for_i,
        cells_in_j,
        coords_for_j,
        time_for_j,
        time_steps,
        num_dim,
        choice,
    ) = params
    for t in time_steps:
        ts_i = np.where(time_for_i == t)[0]
        ts_j = np.where(time_for_j == t)[0]
        if ts_i.size == 0 or ts_j.size == 0:
            continue
        i_weight = ts_i.shape[0] / cells_in_i.shape[0]
        j_weight = ts_j.shape[0] / cells_in_j.shape[0]
        dists = []
        for d in range(num_dim):
            if choice == "wasserstein":
                dists.append(
                    wasserstein_distance(
                        coords_for_i[ts_i][:, d], coords_for_j[ts_j][:, d]
                    )
                )
            elif choice == "energy":
                dists.append(
                    energy_distance(coords_for_i[ts_i][:, d],
                                    coords_for_j[ts_j][:, d])
                )
            else:
                print("not supported")
                sys.exit(2)
        ts_avg = np.mean(dists) * np.mean([i_weight, j_weight])
        dist += ts_avg
    return dist
