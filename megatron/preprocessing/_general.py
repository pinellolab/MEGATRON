"""General preprocessing functions"""

import numpy as np
import pandas as pd
from sklearn.utils import sparsefuncs
from sklearn import preprocessing
from scipy.sparse import (
    issparse,
    csr_matrix,
)

from ._utils import (
    cal_tf_idf
)


def log_transform(adata):
    """Return the natural logarithm of one plus the input array, element-wise.
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.

    Returns
    -------
    updates `adata` with the following fields.
    X: `numpy.ndarray` (`adata.X`)
        Store #observations × #var_genes logarithmized data matrix.
    """

    adata.X = np.log1p(adata.X)
    return None


def binarize(adata,
             threshold=1e-5,
             copy=True):
    """Binarize an array.
    """
    adata.X = preprocessing.binarize(adata.X,
                                     threshold=threshold,
                                     copy=copy)


def normalize(adata, method='lib_size', scale_factor=1e4, save_raw=True):
    """Normalize count matrix.
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    method: `str`, optional (default: 'lib_size')
        Choose from {{'lib_size','tf_idf'}}
        Method used for dimension reduction.
        'lib_size': Total-count normalize (library-size correct)
        'tf_idf': TF-IDF (term frequency–inverse document frequency)
                  transformation
    Returns
    -------
    updates `adata` with the following fields.
    X: `numpy.ndarray` (`adata.X`)
        Store #observations × #var_genes normalized data matrix.
    """
    if(method not in ['lib_size', 'tf_idf']):
        raise ValueError("unrecognized method '%s'" % method)
    if(save_raw):
        adata.layers['raw'] = adata.X.copy()
    if(method == 'lib_size'):
        sparsefuncs.inplace_row_scale(adata.X, 1/adata.X.sum(axis=1).A)
        adata.X = adata.X*scale_factor
    if(method == 'tf_idf'):
        adata.X = cal_tf_idf(adata.X)


def add_clones(adata,
               mat,
               anno=None):
    """Add clonal information into anndata object
    Parameters
    ----------
    adata: `AnnData`
        Annotated data matrix.
    mat: `array_like`
        A cells-by-clones relation matrix.
    anno: `pd.DataFrame`, optional (default: None)
        Annotation of clones.
        If None, annotation dataframe will be auto-generated

    Returns
    -------
    Updates `adata` with the following fields.
    X_clone: `array_like` (`.obsm['X_clone']`)
        Store #cells × #clones relation matrix.

    Updates `adata.uns['clone']` with the following fields.
    anno: `pd.DataFrame` (`.uns['clone']['anno']`)
        Store annotation of clones
    """

    if not issparse(mat):
        mat = csr_matrix(mat)
    if anno is None:
        anno = pd.DataFrame(
            index=np.arange(mat.shape[1]).astype(str))
    assert isinstance(anno, pd.DataFrame),\
        "'anno' must be pd.DataFrame"
    assert mat.shape[1] == anno.shape[0],\
        "clone and its annotation must match"

    adata.obsm['X_clone'] = mat.copy()
    adata.uns['clone'] = dict()
    adata.uns['clone']['anno'] = anno.copy()


def add_clone_traj(adata,
                   mat,
                   anno=None):
    """Add clonal trajectories into anndata object
    Parameters
    ----------
    adata: `AnnData`
        Annotated data matrix.
    mat: `array_like`
        A cells-by-clone_trajectories relation matrix.
    anno: `pd.DataFrame`, optional (default: None)
        Annotation of clone trajectories.
        If None, annotation dataframe will be auto-generated

    Returns
    -------
    Updates `adata` with the following fields.
    X_clone_traj: `numpy.ndarray` (`.obsm['X_clone_traj']`)
        Store #cells × #clones relation matrix.

    Updates `adata.uns['clone_traj']` with the following fields.
    anno: `pd.DataFrame` (`.uns['clone_traj']['anno']`)
        Store annotation of clone trajectories
    """

    if not issparse(mat):
        mat = csr_matrix(mat)
    if anno is None:
        anno = pd.DataFrame(
            index=np.arange(mat.shape[1]).astype(str))
    assert isinstance(anno, pd.DataFrame),\
        "'anno' must be pd.DataFrame"
    assert mat.shape[1] == anno.shape[0],\
        "clone trajectory and its annotation must match"

    adata.obsm['X_clone_traj'] = mat.copy()
    adata.uns['clone_traj'] = dict()
    adata.uns['clone_traj']['anno'] = anno.copy()
