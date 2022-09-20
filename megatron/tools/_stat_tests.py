from multiprocessing.sharedctypes import Value
from scipy.stats import ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.regression.mixed_linear_model import MixedLM


import functools
import pandas as pd
import numpy as np


def _ttest_single_var(adata,
                      gene,
                      clone1=None,
                      clone2=None,
                      layer=None,
                      batch=None,
                      clone_anno='X_clone',
                      anno_type='hierarchical',
                      test='wilcoxon'
                      ):

    if anno_type not in adata.uns['clone']['anno']:
        raise ValueError(
            f"Annotation '{anno_type}' not found in adata.uns['clone']['anno']")

    # get observations and metaclone annotations
    if layer is None:
        X = adata[:, gene].X.copy()
    else:
        X = adata[:, gene].layers[layer].copy()
    # X_norm = (X-X.mean())
    clone_metaclone = adata.uns['clone']['anno'][anno_type]

    # assert same number of clones in obsm and uns['clone']['anno']
    assert (adata.obsm[clone_anno].shape[1] == clone_metaclone.shape[0])

    # get indices of cells belonging to each metaclone
    is_c1 = clone_metaclone.isin(clone1).values
    is_c2 = clone_metaclone.isin(clone2).values

    # get observations
    grp1 = X[(adata.obsm[clone_anno] @ is_c1).astype(bool)]
    grp2 = X[(adata.obsm[clone_anno] @ is_c2).astype(bool)]

    if test == "wilcoxon":
        _, p = mannwhitneyu(grp1, grp2)
        coef = grp1.mean()-grp2.mean()

    elif test == "t-test":
        _, p = ttest_ind(grp1, grp2)
        coef = grp1.mean()-grp2.mean()

    elif test == "lmm":  # EXPERIMENTAL
        batch = "library"
        mat = adata.obsm[clone_anno] @ np.vstack((is_c1, is_c2)).T
        mat = np.float32(mat)
        contrast = mat[:, 0]-mat[:, 1]
        group = adata.obs[batch]
        md = MixedLM(X, contrast, group)
        mdf = md.fit(method=['lbfgs'], warn_convergence=False)
        coef = mdf.fe_params[0]
        p = mdf.pvalues[0]
        return ((coef, p))
    else:
        raise ValueError("Only 't-test' or 'wilcoxon' methods supported")
    return ((float(coef), float(p)))


def differential_test_vars(adata,
                           clone_cluster_1,
                           clone_cluster_2,
                           layer=None,
                           batch=None,
                           test='wilcoxon'):
    """perform a differential expression/accessibility/etc. test
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    clone_cluster_1, clone_cluster_2: iterable, Series, DataFrame or dict
        Two list-like of clone_clusters (found in 
        adata.uns['clone']['anno']['hierarchical']) to serve as 
        groups for one-vs-one differential testing
    layer: `str`
        layer of observations for the test, defaults to adata.X
    batch: `str`
        A vector of labels specifying the groups/batches when test='lmm'
    test: `str`
        The differential test type. Currently supports:
            - "t-test": t-test
            - "wilcoxon": Wilcoxon rank-sum (a.k.a. Mann-Whitney U) test
            - "lmm": Linear Mixed model

    layer: `str`, optional (default: None)
        The layer used for the test
    ...

    Returns
    -------
    Updates adata.obs['uns']['{test}_MEGATRON'] with a pd.DataFrame of test results
    """

    if batch is not None and test != 'lmm':
        raise ValueError(
            "batch annotation only supported with method='lmm' (experimental)")

    if test == 'lmm' and (batch is None or batch not in adata.obs):
        raise ValueError(
            "method='lmm' (experimental) requires batch annotation from adata.obs")

    if 'clone' not in adata.uns:
        raise ValueError("Need to add clone annotations to AnnData object")

    if (layer is not None) and (layer not in adata.layers):
        raise ValueError("layer must be None or contained in adata.layers")

    # run test for all genes
    test_gene_adata = functools.partial(
        _ttest_single_var,
        adata,
        clone1=clone_cluster_1,
        clone2=clone_cluster_2,
        layer=layer,
        test=test
    )

    results = list(map(test_gene_adata, adata.var_names))
    coefs, p_values = zip(*results)
    p_values

    lmm_df = pd.DataFrame({'coef': coefs,
                           'p_value': p_values,
                           'gene': adata.var_names,
                           })

    lmm_df = lmm_df.dropna().sort_values('p_value')

    _, p_value_corr = fdrcorrection(lmm_df['p_value'])

    lmm_df = lmm_df.assign(p_value_corr=p_value_corr)

    adata.uns[f'{test}_MEGATRON'] = lmm_df
