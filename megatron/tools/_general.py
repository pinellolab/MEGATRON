"""General-purpose tools"""

from scipy.cluster.hierarchy import (
    dendrogram, linkage, fcluster)


def cluster_clones(adata,
                   n_clusters,
                   method='hierarchical',
                   **kwargs):
    """Cluster clones
    Parameters
    ----------
    Returns
    -------
    """
    list_dist = adata.uns['clone']['distance']
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
    adata.uns['clone']['anno'][method] = clusters.astype(str)
