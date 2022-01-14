import numpy as np
import sys
from sklearn.neighbors import radius_neighbors_graph, kneighbors_graph
from scipy.spatial.distance import squareform


def _mnn(
    mat_clone,
    mat_coord,
    df_time,
    dist="kneighbors",
    radius=1.0,
    neighbors=5,
    mode="distance",
):
    time_steps = np.unique(df_time)
    for t in time_steps:
        print(t)
    print("%s time points" % time_steps)
    num_clones = mat_clone.shape[1]

    if dist == "kneighbors":
        rng = kneighbors_graph(
            mat_coord, neighbors, mode=mode, include_self=True
        )
        print("K-neighbors graph created")
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

    distance_matrix = np.zeros((num_clones, num_clones))
    for i in range(num_clones):

        # get the coords for all cells in i
        cells_in_i = mat_clone[:, i].nonzero()[0]
        # coords_for_i = mat_coord[cells_in_i]
        time_for_i = df_time[cells_in_i]

        ts_i_dict = {}
        for t in time_steps:
            ts_i = cells_in_i[np.where(time_for_i == t)[0]]
            # ts_i_neighbors = coords[(np.isin(coords, ts_i)[:, 0] == True)]
            ts_i_neighbors = coords[(np.isin(coords, ts_i)[:, 0] is True)]
            total_all_ts_i_neighbors = rng[
                ts_i_neighbors[:, 0], ts_i_neighbors[:, 1]
            ].sum()
            ts_i_dict[t] = [ts_i_neighbors, total_all_ts_i_neighbors]

        for j in range(i):
            dist = 0

            # get cells in clone j
            cells_in_j = mat_clone[:, j].nonzero()[0]
            # coords_for_j = mat_coord[cells_in_j]
            time_for_j = df_time[cells_in_j]
            print(".....analyzed %s/%s clone's neighbors" % (i, j))

            ts_dists = []
            for t in time_steps:
                print(".....time step %s" % t)

                ts_i_neighbors, total_all_ts_i_neighbors = ts_i_dict[t]
                # print(ts_i_neighbors, total_all_ts_i_neighbors)
                print(ts_i_neighbors.shape)
                ts_j = cells_in_j[np.where(time_for_j == t)[0]]
                ts_j_neighbors = coords[(np.isin(coords, ts_j)[:, 0] is True)]
                total_all_ts_j_neighbors = rng[
                    ts_j_neighbors[:, 0], ts_j_neighbors[:, 1]
                ].sum()
                # print(ts_j_neighbors, total_all_ts_j_neighbors)
                print(ts_j_neighbors.shape)

                ts_i_neighbors_in_j = ts_i_neighbors[
                    (np.isin(ts_i_neighbors, cells_in_j)[:, 1] is True)
                ]
                total_ts_i_neighbors_in_j = rng[
                    ts_i_neighbors_in_j[:, 0], ts_i_neighbors_in_j[:, 1]
                ].sum()

                ts_j_neighbors_in_i = ts_j_neighbors[
                    (np.isin(ts_j_neighbors, cells_in_i)[:, 1] is True)
                ]
                total_ts_j_neighbors_in_i = rng[
                    ts_j_neighbors_in_i[:, 0], ts_j_neighbors_in_i[:, 1]
                ].sum()
                print(total_all_ts_i_neighbors)
                print(total_all_ts_j_neighbors)
                if (
                    total_all_ts_i_neighbors == 0
                    or total_all_ts_j_neighbors == 0
                ):
                    continue

                ts_i_frac = total_all_ts_i_neighbors / (
                    total_all_ts_i_neighbors + total_ts_i_neighbors_in_j
                )
                print(ts_i_frac)

                ts_j_frac = total_all_ts_j_neighbors / (
                    total_all_ts_j_neighbors + total_ts_j_neighbors_in_i
                )
                print(ts_j_frac)

                ts_dists.append(ts_i_frac)
                ts_dists.append(ts_j_frac)
            print(".....analyzed %s/%s neighborhood across time" % (i, j))
            print(".....%s" % ts_dists)
            if len(ts_dists) == 0:
                print("O MY FUCKING GOD LOOK HERE YOU TWAT")
                print(ts_dists)
                print("O MY FUCKING GOD LOOK HERE YOU TWAT")
            print("MEAN: %s" % np.mean(ts_dists))
            print(".....got mean")
            if i == 130:
                sys.exit()
            # distance_matrix[i][j] += np.mean(ts_dists)
    return squareform(distance_matrix + distance_matrix.transpose())
