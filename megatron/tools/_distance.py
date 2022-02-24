"""Functions to calculate the distances between clones"""

import time
import anndata as ad
import pandas as pd

from ._geodesic import _average_geodesic

# from ._directed_graph import _directed_graph
from ._wasserstein import _wasserstein
from ._mnn import _mnn
from ._centroid import _centroid


def _dist(
    adata,
    target="clone",
    method="geodesic",
    obsm=None,
    layer=None,
    anno_time=None,
    n_jobs=1,
    **kwargs,
):
    """Calculate distances between clones or clone trajectories"""

    if sum(list(map(lambda x: x is not None, [layer, obsm]))) == 2:
        raise ValueError("Only one of `layer` and `obsm` can be used")
    elif obsm is not None:
        if obsm in adata.obsm_keys():
            mat_coord = adata.obsm[obsm]
        else:
            raise ValueError(f"could not find {obsm} in `adata.obsm_keys()`")
    elif layer is not None:
        if layer in adata.layers.keys():
            mat_coord = adata.layers[layer]
        else:
            raise ValueError(
                f"could not find {layer} in `adata.layers.keys()`"
            )
    else:
        mat_coord = adata.X

    if anno_time is None:
        anno_time = 'unknown'
        df_time = pd.DataFrame(
            data=0,
            index=adata.obs_names,
            columns=[anno_time])
    else:
        if anno_time in adata.obs_keys():
            df_time = adata.obs[anno_time].copy()
        else:
            raise ValueError(
                f"could not find {anno_time} in `adata.obs_keys()`")
    mat_clone = adata.obsm[f"X_{target}"]

    ad_input = ad.AnnData(
        X=mat_clone.copy(),
        obs=pd.DataFrame(df_time.copy()),
        var=pd.DataFrame(adata.uns[target]["anno"].copy()),
    )
    ad_input.obsm["X_coord"] = mat_coord.copy()
    ad_input.uns["params"] = {"anno_time": anno_time}

    if method == "geodesic":
        mat_dist = _average_geodesic(ad_input, n_jobs=n_jobs, **kwargs)
    # elif method == 'directed_graph':
    #     mat_dist = _directed_graph(mat_clone,
    #                                mat_coord,
    #                                df_time,
    #                                n_jobs=n_jobs,
    #                                **kwargs)
    elif method == "mnn":
        mat_dist = _mnn(mat_clone, mat_coord, df_time, n_jobs=n_jobs, **kwargs)
    elif method == "wasserstein":
        mat_dist = _wasserstein(
            mat_clone, mat_coord, df_time, n_jobs=n_jobs, **kwargs
        )
    elif method == "centroid":
        mat_dist = _centroid(
            mat_clone,
            mat_coord,
        )
    else:
        raise ValueError(f"unrecognized method {method}")
    adata.uns[target][f"distance_{method}"] = mat_dist
    adata.uns[target]["distance"] = mat_dist


def _set_dist(adata, target="clone", method="geodesic"):
    """Choose from calculated distance and set it to the current distance"""
    assert method in [
        "geodesic",
        "mnn",
        "wasserstein",
        "sinkhorn",
    ], f"unrecognized method {method}"
    if f"distance_{method}" in adata.uns[target].keys():
        adata.uns[target]["distance"] = adata.uns[target][
            f"distance_{method}"
        ].copy()
    else:
        raise ValueError(f'"{method}" has not been used yet')


def clone_distance(
    adata,
    method="geodesic",
    obsm=None,
    layer=None,
    anno_time=None,
    n_jobs=1,
    **kwargs,
):
    """Calculate distances between clones

    Parameters
    ----------
    adata: `AnnData`
        Anndata object.
    method: `str`, (default: 'directed_graph');
        Method used to calculate clonal distances.
        Possible methods:
        - 'geodesic': graph-based geodesic distance
        - 'directed_graph': shortest-path-based directed graph
        - 'mnn':
        - 'wasserstein':
        - 'centroid'
    layer: `str`, optional (default: None)
        The layer used to perform UMAP
    obsm: `str`, optional (default: None)
        The multi-dimensional annotation of observations used to perform UMAP
    time: `str`, optional (default: 'time')
        Column name of observations (adata.obs) indicating temporal information
    n_jobs: `int`, optional (default: 1)
        The number of parallel jobs to run
    **kwargs:
        Additional arguments to each method
        - 'geodesic':
            use_weight: `bool`, optional (default: False)
                Use weights for time annotation
                Only valid when 'geodesic' is used.
            weight_time: `dict`, optional (default: None)
                a dictionary of weights for time annotation
                Only valid when 'geodesic' is used.

    Returns
    -------
    updates `adata.uns['clone']` with the following field.
    distance: `sparse matrix`` (`.uns['clone']['distance']`)
        A condensed clone distance matrix.
        It can be converted into a redundant square matrix using `squareform`
        from Scipy.
    """

    st = time.time()
    _dist(
        adata,
        target="clone",
        method=method,
        obsm=obsm,
        layer=layer,
        anno_time=anno_time,
        n_jobs=n_jobs,
        **kwargs,
    )
    ed = time.time()
    print(f"Finished: {(ed-st)/60} mins")


def set_clone_distance(adata, method="directed_graph"):
    """Set the current distance matrix to the one calculated
    by the specified method

    Parameters
    ----------
    adata: `AnnData`
        Anndata object.
    method: `str`, (default: 'directed_graph');
        Method used to calculate clonal distances.
        Possible methods:
        - 'geodesic': graph-based geodesic distance
        - 'directed_graph': shortest-path-based directed graph
        - 'mnn':
        - 'wasserstein':
        - 'centroid'

    Returns
    -------
    updates `adata.uns['clone_traj']` with the following field.
    distance: `sparse matrix`` (`.uns['clone_traj']['distance']`)
        A condensed clone distance matrix.
        It can be converted into a redundant square matrix using `squareform`
        from Scipy.
    """
    _set_dist(adata, target="clone", method=method)
