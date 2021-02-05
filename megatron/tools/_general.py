"""General-purpose tools"""

from scipy.cluster.hierarchy import (
    dendrogram, linkage, fcluster)


def _cluster(adata,
             n_clusters,
             target='clone',
             method='hierarchical',
             **kwargs):
    """Cluster clones or clone trajectories
    """
    list_dist = adata.uns[target]['distance']
    if method == 'hierarchical':
        Z = linkage(list_dist,
                    'ward',
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
             target='clone',
             **kwargs)


def cluster_clone_traj(adata,
                       n_clusters=2,
                       method='hierarchical',
                       **kwargs):
    """Cluster clone paths

    Parameters
    ----------

    Returns
    -------
    """
    _cluster(adata,
             n_clusters=n_clusters,
             method=method,
             target='clone_traj',
             **kwargs)
