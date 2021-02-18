"""Functions to calculate the distances between clones"""

import time

from ._directed_graph import _directed_graph
from ._wasserstein import _wasserstein
from ._mnn import _mnn
from ._sinkhorn import _sinkhorn


def _dist(adata,
          target='clone',
          method='directed_graph',
          obsm=None,
          layer=None,
          anno_time='time',
          **kwargs):
    """Calculate distances between clones or clone trajectories
    """

    if(sum(list(map(lambda x: x is not None,
                    [layer, obsm]))) == 2):
        raise ValueError("Only one of `layer` and `obsm` can be used")
    elif obsm is not None:
        if obsm in adata.obsm_keys():
            mat_coord = adata.obsm[obsm]
        else:
            raise ValueError(
                f'could not find {obsm} in `adata.obsm_keys()`')
    elif layer is not None:
        if layer in adata.layers.keys():
            mat_coord = adata.layers[layer]
        else:
            raise ValueError(
                f'could not find {layer} in `adata.layers.keys()`')
    else:
        mat_coord = adata.X

    if anno_time in adata.obs_keys():
        df_time = adata.obs[anno_time].copy()
    else:
        raise ValueError(
            f'could not find {anno_time} in `adata.obs_keys()`')
    mat_clone = adata.obsm[f'X_{target}']

    if method == 'directed_graph':
        mat_dist = _directed_graph(mat_clone,
                                   mat_coord,
                                   df_time,
                                   **kwargs)
    elif method == 'mnn':
        mat_dist = _mnn(mat_clone,
                        mat_coord,
                        df_time,
                        **kwargs)
    elif method == 'wasserstein':
        mat_dist = _wasserstein(mat_clone,
                                mat_coord,
                                df_time,
                                **kwargs)
    elif method == 'sinkhorn':
        mat_dist = _sinkhorn(mat_clone,
                             mat_coord,
                             df_time,
                             **kwargs)
    else:
        raise ValueError(
            f'unrecognized method {method}')
    adata.uns[target]['distance'] = mat_dist


def clone_distance(adata,
                   method='directed_graph',
                   obsm=None,
                   layer=None,
                   anno_time='time',
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
        - 'directed_graph': shortest-path-based directed graph
        - 'mnn':
        - 'wasserstein'
    layer: `str`, optional (default: None)
        The layer used to perform UMAP
    obsm: `str`, optional (default: None)
        The multi-dimensional annotation of observations used to perform UMAP
    time: `str`, optional (default: 'time')
        Column name of observations (adata.obs) indicating temporal information
    **kwargs:
        Additional arguments to each method

    Returns
    -------
    updates `adata.uns['clone']` with the following field.
    distance: `sparse matrix`` (`.uns['clone']['distance']`)
        A condensed clone distance matrix.
        It can be converted into a redundant square matrix using `squareform`
        from Scipy.
    """

    st = time.time()
    _dist(adata,
          target='clone',
          method=method,
          obsm=obsm,
          layer=layer,
          anno_time=anno_time,
          **kwargs)
    ed = time.time()
    print(f'Finished: {(ed-st)/60} mins')


def clone_traj_distance(adata,
                        method='directed_graph',
                        obsm=None,
                        layer=None,
                        anno_time='time',
                        **kwargs,
                        ):
    """Calculate distances between clone paths

    Parameters
    ----------
    adata: `AnnData`
        Anndata object.
    method: `str`, (default: 'directed_graph');
        Method used to calculate clonal distances.
        Possible methods:
        - 'directed_graph': shortest-path-based directed graph
        - 'mnn':
        - 'wasserstein'
    layer: `str`, optional (default: None)
        The layer used to perform UMAP
    obsm: `str`, optional (default: None)
        The multi-dimensional annotation of observations used to perform UMAP
    time: `str`, optional (default: 'time')
        Column name of observations (adata.obs) indicating temporal information
    **kwargs:
        Additional arguments to each method

    Returns
    -------
    updates `adata.uns['clone_traj']` with the following field.
    distance: `sparse matrix`` (`.uns['clone_traj']['distance']`)
        A condensed clone distance matrix.
        It can be converted into a redundant square matrix using `squareform`
        from Scipy.
    """

    st = time.time()
    _dist(adata,
          target='clone_traj',
          method=method,
          obsm=obsm,
          layer=layer,
          anno_time=anno_time,
          **kwargs)
    ed = time.time()
    print(f'Finished: {(ed-st)/60} mins')