"""Utility functions and classes"""

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from pandas.api.types import (
    is_numeric_dtype,
    is_string_dtype,
    is_categorical_dtype,
)
import matplotlib as mpl

from ._palettes import (
    default_20,
    default_28,
    default_102
)


def get_colors(arr,
               vmin=None,
               vmax=None,
               clip=False):
    """Generate a list of colors for a given array
    """

    if not isinstance(arr, (pd.Series, np.ndarray)):
        raise TypeError("`arr` must be pd.Series or np.ndarray")
    colors = []
    if is_numeric_dtype(arr):
        image_cmap = mpl.rcParams['image.cmap']
        cm = mpl.cm.get_cmap(image_cmap, 512)
        if vmin is None:
            vmin = min(arr)
        if vmax is None:
            vmax = max(arr)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=clip)
        colors = [mpl.colors.to_hex(cm(norm(x))) for x in arr]
    elif is_string_dtype(arr) or is_categorical_dtype(arr):
        categories = np.unique(arr)
        length = len(categories)
        # check if default matplotlib palette has enough colors
        # mpl.style.use('default')
        if len(mpl.rcParams['axes.prop_cycle'].by_key()['color']) >= length:
            cc = mpl.rcParams['axes.prop_cycle']()
            palette = [mpl.colors.rgb2hex(next(cc)['color'])
                       for _ in range(length)]
        else:
            if length <= 20:
                palette = default_20
            elif length <= 28:
                palette = default_28
            elif length <= len(default_102):  # 103 colors
                palette = default_102
            else:
                rgb_rainbow = mpl.cm.rainbow(np.linspace(0, 1, length))
                palette = [mpl.colors.rgb2hex(rgb_rainbow[i, :-1])
                           for i in range(length)]
        colors = pd.Series(['']*len(arr))
        for i, x in enumerate(categories):
            ids = np.where(arr == x)[0]
            colors[ids] = palette[i]
        colors = list(colors)
    else:
        raise TypeError("unsupported data type for `arr`")
    return colors


def generate_palette(arr):
    """Generate a color palette for a given array
    """

    if not isinstance(arr, (pd.Series, np.ndarray)):
        raise TypeError("`arr` must be pd.Series or np.ndarray")
    colors = []
    if is_string_dtype(arr) or is_categorical_dtype(arr):
        categories = np.unique(arr)
        length = len(categories)
        # check if default matplotlib palette has enough colors
        # mpl.style.use('default')
        if len(mpl.rcParams['axes.prop_cycle'].by_key()['color']) >= length:
            cc = mpl.rcParams['axes.prop_cycle']()
            palette = [mpl.colors.rgb2hex(next(cc)['color'])
                       for _ in range(length)]
        else:
            if length <= 20:
                palette = default_20
            elif length <= 28:
                palette = default_28
            elif length <= len(default_102):  # 103 colors
                palette = default_102
            else:
                rgb_rainbow = mpl.cm.rainbow(np.linspace(0, 1, length))
                palette = [mpl.colors.rgb2hex(rgb_rainbow[i, :-1])
                           for i in range(length)]
        colors = pd.Series(['']*len(arr))
        for i, x in enumerate(categories):
            ids = np.where(arr == x)[0]
            colors[ids] = palette[i]
        colors = list(colors)
    else:
        raise TypeError("unsupported data type for `arr`")
    dict_palette = dict(zip(arr, colors))
    return dict_palette


def get_vars_by_metaclone(adata, var_subset=None, metaclone_subset=None, force=False):
    """
    Return a "long" pandas DataFrame of observations for each cell 
    with metaclone information attached.
    """

    if var_subset is None and metaclone_subset is None and not force:
        raise Exception("Provide one of var_subset or metaclone_subset, or set 'force=True' to return a df for all vars + cells.")
    
    if var_subset is None:
            var_subset = adata.var_names

    metaclones = sorted(pd.unique(adata.uns['clone']['anno']['hierarchical']))

    if metaclone_subset is None:
        metaclone_subset = metaclones
    
    clone_x_metaclone = pd.get_dummies(adata.uns['clone']['anno']).values
    cell_x_clone = adata.obsm['X_clone']
    cell_x_metaclone = (cell_x_clone @ clone_x_metaclone).astype(bool)
    n_cells, n_metaclones = cell_x_metaclone.shape
    
    def _get_metaclone_observations(j):
        if metaclones[j] not in metaclone_subset:
            return None
        
        adata_subset = adata[:,var_subset][cell_x_metaclone[:,j],:]
        observations = adata_subset.X.copy()
        if issparse(observations):
            observations = observations.todense()
        subset_df = pd.DataFrame(np.asarray(observations), columns=list(var_subset))
        subset_df = subset_df.assign(metaclone=metaclones[j])
        if 'pseudotime' in adata_subset.obs_keys():
            subset_df=subset_df.assign(pseudotime=adata_subset.obs['pseudotime'].values)
        # subset_df=subset_df.melt(id_vars=['pseudotime'], var_name='gene').assign(metaclone=metaclones[j])
        return subset_df

    return pd.concat(map(_get_metaclone_observations, range(n_metaclones))).reset_index(drop=True)
