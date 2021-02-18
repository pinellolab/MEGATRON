import numpy as np
from scipy.spatial.distance import squareform
import geomloss
import torch

def _sinkhorn(mat_clone,
              mat_coord,
              df_time,
              p=2,
              blur=0.5,
              scaling=0.8,
              mode="distance"):
    Loss =  geomloss.SamplesLoss("sinkhorn", p=p, blur=blur, scaling=scaling)
    time_steps = np.unique(df_time)
    num_clones = mat_clone.shape[1]
    num_dim = mat_coord.shape[1]
    distance_matrix = np.zeros((num_clones, num_clones))
    num_noninform = 0
    ctr=0
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
                ts_i = np.where(time_for_i == t)[0]
                ts_j = np.where(time_for_j == t)[0]

                # continue if we can't do anything
                if ts_i.size == 0 or ts_j.size == 0:
                    continue
                i_weight = ts_i.shape[0] / cells_in_i.shape[0]
                j_weight = ts_j.shape[0] / cells_in_j.shape[0]
                dists = []
                #print(coords_for_i[ts_i])
                Wass_xy = Loss(torch.from_numpy(coords_for_i[ts_i]),
                                        torch.from_numpy(coords_for_j[ts_j]))
                dists.append(Wass_xy)
                #for d in range(num_dim):
                #    dists.append(Loss(coords_for_i[ts_i][:,d], coords_for_j[ts_j][:,d]))
                ts_avg = np.mean(dists) * np.mean([i_weight, j_weight])
                dist += ts_avg
            if dist == 0:
                #print(time_for_j)
                #print(time_for_i)
                #L00k -> THESE VALUES SHOULD BE INFINITY
                num_noninform += 1
            distance_matrix[i][j] = dist
            ctr += 1
            if ctr % 10000 == 0: print("%s analyzed" % ctr)
    #print("Out of " + str(math.comb(num_clones, 2)) + " clonal distances, " + str(num_noninform) + " are noninformative")
    return squareform(distance_matrix + distance_matrix.transpose())
