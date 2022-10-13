import megatron as me
from scipy.sparse import load_npz
import pytest


@pytest.fixture
def adata():
    return me.read_h5ad(
        "tests/data/chromVAR_lareau_2021.h5ad")


@pytest.fixture
def mat_clones():
    return load_npz(
        "tests/data/clones_lareau_2021.npz")


def test_cluster_clones_lareau_2021(adata, mat_clones, tmp_path):
    me.settings.set_workdir(tmp_path / "result_lareau21")
    me.settings.set_figure_params(dpi=80,
                                  style='white',
                                  fig_size=[5, 5],
                                  rc={'image.cmap': 'viridis'})

    me.pp.add_clones(adata, mat=mat_clones)
    me.pp.filter_clones(adata, min_cells=1)
    me.pl.scatter(adata, color=['new_name', 'library',
                  'mtDNAcoverage'], drawing_order='random')
    me.pl.umap(adata, color=adata.var_names[:3], layer='z_norm')
    me.tl.clone_distance(adata,
                         obsm='X_umap',
                         anno_time=None,
                         method='geodesic',
                         n_jobs=8)
    me.tl.cluster_clones(adata,
                         n_clusters=3,
                         method='hierarchical')
    me.pl.clone_clusters(adata,
                         group='hierarchical',
                         obsm='X_umap',
                         show_contour=True,
                         levels=5,
                         thresh=0.2)

    # build k-NN graph and inspect composition by metaclone
    me.tl.build_graph(adata, obsm='X_umap', k=3, n_clusters=40)
    me.pl.cluster_graph(adata, obsm='X_umap', node_color="black", alpha=0.8)
    me.pl.cluster_pie_graph(adata, obsm="X_umap")

    # extract graph-based pseudotime
    progenitor_clusters = [20]
    me.tl.calculate_pseudotime(adata, progenitor_clusters)
    adata.obs['pseudotime'] = adata.uns['cluster_pseudotime'].loc[adata.obs['cluster']].values
    me.pl.umap(adata, color=['pseudotime'])

    # differential testing by metaclone
    group1 = ['1']
    group2 = ['2']
    me.tl.differential_test_vars(adata, group1, group2, test="t-test")
    me.tl.differential_test_vars(adata, group1, group2, test="wilcoxon")
