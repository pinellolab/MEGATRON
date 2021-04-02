import numpy as np
from scipy.stats import wasserstein_distance, energy_distance
from scipy.spatial.distance import squareform

def _wasserstein(mat_clone,
                 mat_coord,
                 df_time,
                 choice="wasserstein"):
    time_steps = np.unique(df_time)
    num_clones = mat_clone.shape[1]
    num_dim = mat_coord.shape[1]
    distance_matrix = np.zeros((num_clones, num_clones))
    num_noninform = 0
    for i in range(num_clones):
        cells_in_i = mat_clone[:,i].nonzero()[0]
        coords_for_i = mat_coord[cells_in_i]
        time_for_i = df_time[cells_in_i]
        for j in range(i):
            dist = 0
            cells_in_j = mat_clone[:,j].nonzero()[0]
            coords_for_j = mat_coord[cells_in_j]
            time_for_j = df_time[cells_in_j]
            for t in time_steps:
                #print(t)
                #print(time_for_j)
                #print(time_for_i)
                ts_i = np.where(time_for_i == t)[0]
                ts_j = np.where(time_for_j == t)[0]
                #print(ts_i.size)
                #print(ts_j.size)
                # continue if we can't do anything
                if ts_i.size == 0 or ts_j.size == 0:
                    continue
                i_weight = ts_i.shape[0] / cells_in_i.shape[0]
                j_weight = ts_j.shape[0] / cells_in_j.shape[0]
                dists = []
                for d in range(num_dim):
                    if choice == "wasserstein":
                        dists.append(wasserstein_distance(coords_for_i[ts_i][:,d], coords_for_j[ts_j][:,d]))
                    elif choice == "energy":
                        dists.append(energy_distance(coords_for_i[ts_i][:,d], coords_for_j[ts_j][:,d]))
                    else:
                        print("not supported")
                        sys.exit(2)
                ts_avg = np.mean(dists) * np.mean([i_weight, j_weight])
                dist += ts_avg
            if dist == 0:
                #print(time_for_j)
                #print(time_for_i)
                #L00k -> THESE VALUES SHOULD BE INFINITY
                num_noninform += 1
            distance_matrix[i][j] = dist
    #print("Out of " + str(math.comb(num_clones, 2)) + " clonal distances, " + str(num_noninform) + " are noninformative")
    #print(squareform(distance_matrix + distance_matrix.transpose()))
    return squareform(distance_matrix + distance_matrix.transpose())
