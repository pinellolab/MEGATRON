"""General-purpose tools"""

import numpy as np
import copy
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform


def _cluster(adata,
             n_clusters,
             target='clone',
             method='hierarchical',
             linkage='ward',
             **kwargs):
    """Cluster clones or clone trajectories
    """
    list_dist = adata.uns[target]['distance']
    if method == 'hierarchical':
        Z = scipy_linkage(list_dist,
                          method=linkage,
                          **kwargs)
        clusters = fcluster(Z,
                            n_clusters,
                            criterion='maxclust',
                            )
    else:
        raise ValueError(
            f'unrecognized method `{method}`')
    adata.uns[target]['anno'][method] = clusters.astype(str)


def cluster_clones(adata,
                   n_clusters=2,
                   method='hierarchical',
                   linkage='ward',
                   **kwargs):
    """Cluster clones

    Parameters
    ----------
    Returns
    -------
    """
    _cluster(adata,
             n_clusters=n_clusters,
             method=method,
             linkage=linkage,
             target='clone',
             **kwargs)


def subset_clones(
    adata,
    anno_value,
    anno_col='hierarchical',
):
    """Subset clones

    Parameters
    ----------
    adata: `AnnData`
        Annotated data matrix.
    anno_value: `list`
        The annotation value(s) used for selecting clones
    anno_col:
        The column name of clonal annotation used for selecting clones
    Returns
    -------
    A new `adata` containing cells belonging to the subset of clones
    """
    if anno_col is None:
        print("Please specify the column of clonal annotation `anno_col`")
    else:
        if anno_col in adata.uns['clone']['anno'].columns:
            mat_clone = adata.obsm['X_clone']
            id_clones_x = np.where(
                np.isin(
                    adata.uns['clone']['anno'][anno_col],
                    list(anno_value)))[0]
            id_cells_x = np.where(
                adata.obsm['X_clone'][:, id_clones_x].sum(axis=1).A1 > 0)[0]
            adata_subset = adata[id_cells_x, ].copy()
            adata_subset.obsm['X_clone'] = \
                mat_clone[id_cells_x, :][:, id_clones_x].copy()
            adata_subset.uns['clone'] = copy.deepcopy(adata.uns['clone'])
            for xx in adata_subset.uns['clone'].keys():
                if xx == 'anno':
                    anno_clone = adata.uns['clone']['anno']
                    adata_subset.uns['clone']['anno'] = \
                        anno_clone.iloc[id_clones_x, ].copy()
                if xx.startswith('distance'):
                    dist_subset = \
                        squareform(
                            squareform(adata.uns['clone'][xx])
                            [id_clones_x, :][:, id_clones_x])
                    adata_subset.uns['clone'][xx] = dist_subset.copy()
        else:
            raise ValueError(
                f'could not find "{anno_col}" in '
                '`adata.uns["clone"]["anno"].columns`')
    return adata_subset
