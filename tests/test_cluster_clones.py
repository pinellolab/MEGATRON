import megatron as me
from scipy.sparse import load_npz
import pytest


@pytest.fixture
def adata():
    return me.read_h5ad(
        "tests/data/rnaseq_weinreb20.h5ad")


@pytest.fixture
def mat_clones():
    return load_npz(
        "tests/data/clones_weinreb20.npz")


def test_cluster_clones_weinreb20(adata, mat_clones, tmp_path):
    me.settings.set_workdir(tmp_path / "result_weinreb20")
    me.settings.set_figure_params(dpi=80,
                                  style='white',
                                  fig_size=[5, 5],
                                  rc={'image.cmap': 'viridis'})
    me.pp.add_clones(adata,
                     mat=mat_clones)
    me.pp.filter_cells_rna(adata, min_n_genes=5)
    me.pp.filter_genes(adata, min_n_cells=2)
    me.pp.cal_qc_rna(adata)
    me.pl.violin(adata,
                 list_obs=['n_counts', 'n_genes', 'pct_mt'],
                 alpha=0.3)
    me.pp.normalize(adata, method='lib_size')
    me.pp.log_transform(adata)
    me.pp.select_variable_genes(adata,
                                n_top_genes=60)
    me.pl.variable_genes(adata, show_texts=False)
    me.pp.pca(adata,
              feature='highly_variable',
              n_components=50)
    me.pl.pca_variance_ratio(adata,
                             show_cutoff=False)
    me.pp.select_pcs(adata, n_pcs=50)
    me.pl.pca_variance_ratio(adata)
    me.tl.umap(adata, obsm='X_pca', n_dim=50)
    adata.obs['Time point'] = adata.obs['Time point'].astype(str)
    me.pl.umap(adata,
               color=['Time point', 'Population', 'Annotation'],
               drawing_order='random')
    me.pp.filter_clones(adata, min_cells=1)
    me.tl.clone_distance(adata,
                         method='directed_graph',
                         obsm='X_SPRING',
                         layer=None,
                         anno_time='Time point',
                         radius=500)
    me.tl.clone_distance(adata,
                         method='mnn',
                         obsm='X_SPRING',
                         layer=None,
                         anno_time='Time point')
    me.tl.clone_distance(adata,
                         method='wasserstein',
                         obsm='X_SPRING',
                         layer=None,
                         anno_time='Time point')
    adata.uns['clone']['distance'] = \
        adata.uns['clone']['distance_directed_graph'].copy()
    me.tl.cluster_clones(adata,
                         n_clusters=3,
                         method='hierarchical')
    adata.uns['clone']['distance'] = \
        adata.uns['clone']['distance_mnn'].copy()
    me.tl.cluster_clones(adata,
                         n_clusters=3,
                         method='hierarchical')
    adata.uns['clone']['distance'] = \
        adata.uns['clone']['distance_wasserstein'].copy()
    me.tl.cluster_clones(adata,
                         n_clusters=3,
                         method='hierarchical')
    me.pl.clone_scatter(adata,
                        group='hierarchical',
                        obsm='X_SPRING',
                        show_contour=True,
                        levels=6,
                        thresh=0.1)
