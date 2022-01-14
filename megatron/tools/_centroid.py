import numpy as np
from scipy.spatial.distance import squareform


def _centroid(mat_clone, mat_coord, df_time):
    num_clones = mat_clone.shape[1]
    distance_matrix = np.zeros((num_clones, num_clones))
    for i in range(num_clones):
        cells_in_i = mat_clone[:, i].nonzero()[0]
        coords_for_i = mat_coord[cells_in_i]
        i_centroid = np.mean(coords_for_i, axis=0)
        for j in range(i):
            cells_in_j = mat_clone[:, j].nonzero()[0]
            coords_for_j = mat_coord[cells_in_j]
            j_centroid = np.mean(coords_for_j, axis=0)
            distance_matrix[i][j] = np.linalg.norm(i_centroid - j_centroid)
    return squareform(distance_matrix + distance_matrix.transpose())
