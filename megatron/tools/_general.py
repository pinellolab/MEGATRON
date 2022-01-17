"""General-purpose tools"""

from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.cluster.hierarchy import fcluster


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


def cluster_clone_traj(adata,
                       n_clusters=2,
                       method='hierarchical',
                       linkage='ward',
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
             linkage=linkage,
             target='clone_traj',
             **kwargs)
