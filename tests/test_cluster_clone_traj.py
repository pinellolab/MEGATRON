import megatron as me
from scipy.sparse import load_npz
import pytest


@pytest.fixture
def adata():
    return me.read_h5ad(
        "tests/data/rnaseq_biddy18.h5ad")


@pytest.fixture
def mat_clone_traj():
    return load_npz(
        "tests/data/clone_traj_biddy18.npz")


def test_cluster_clones_biddy18(adata, mat_clone_traj, tmp_path):
    me.settings.set_workdir(tmp_path / "result_biddy18")
    me.settings.set_figure_params(dpi=80,
                                  style='white',
                                  fig_size=[5, 5],
                                  rc={'image.cmap': 'viridis'})
    me.pp.add_clone_traj(adata,
                         mat=mat_clone_traj)
    me.pp.filter_cells_rna(adata, min_n_genes=5)
    me.pp.filter_genes(adata, min_n_cells=2)
    me.pp.cal_qc_rna(adata)
    me.pl.violin(adata,
                 list_obs=['n_counts', 'n_genes', 'pct_mt'],
                 alpha=0.3)
    me.pp.normalize(adata, method='lib_size')
    me.pp.log_transform(adata)
    me.pp.select_variable_genes(adata,
                                n_top_genes=100)
    me.pl.variable_genes(adata, show_texts=False)
    me.pp.pca(adata,
              feature='highly_variable',
              n_components=50)
    me.pl.pca_variance_ratio(adata,
                             show_cutoff=False)
    me.pp.select_pcs(adata, n_pcs=30)
    me.pl.pca_variance_ratio(adata)
    me.tl.umap(adata, obsm='X_pca', n_dim=30)
    me.pl.umap(adata,
               color=['Timepoint', 'Cluster.Seurat', 'n_genes'],
               drawing_order='random')
    me.pp.filter_clone_traj(adata, min_cells=1)
    me.tl.clone_traj_distance(adata,
                              method='geodesic',
                              obsm='X_tsne_paper',
                              layer=None,
                              anno_time='Day')
    me.tl.clone_traj_distance(adata,
                              method='mnn',
                              obsm='X_tsne_paper',
                              layer=None,
                              anno_time='Day')
    me.tl.clone_traj_distance(adata,
                              method='wasserstein',
                              obsm='X_tsne_paper',
                              layer=None,
                              anno_time='Day')

    adata.uns['clone_traj']['distance'] = \
        adata.uns['clone_traj']['distance_geodesic'].copy()
    me.tl.cluster_clone_traj(adata,
                             n_clusters=3,
                             method='hierarchical')
    adata.uns['clone_traj']['distance'] = \
        adata.uns['clone_traj']['distance_mnn'].copy()
    me.tl.cluster_clone_traj(adata,
                             n_clusters=3,
                             method='hierarchical')
    adata.uns['clone_traj']['distance'] = \
        adata.uns['clone_traj']['distance_wasserstein'].copy()
    me.tl.cluster_clone_traj(adata,
                             n_clusters=3,
                             method='hierarchical')
    me.pl.clone_traj_scatter(adata,
                             group='hierarchical',
                             obsm='X_tsne_paper',
                             show_contour=True,
                             levels=6,
                             thresh=0.1)
