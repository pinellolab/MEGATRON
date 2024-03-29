"""plotting functions"""

from cmath import inf
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
from pandas.core.dtypes.common import is_numeric_dtype
import seaborn as sns
from adjustText import adjust_text
from pandas.api.types import (
    is_string_dtype,
    is_categorical_dtype,
)
from scipy.cluster.hierarchy import (
    dendrogram,
    linkage
)
import plotly.express as px
# import plotly.graph_objects as go


from .._settings import settings
from ._utils import (
    generate_palette,
    get_vars_by_metaclone
)
from ..tools._geodesic import build_graph


def violin(adata,
           list_obs=None,
           list_var=None,
           jitter=0.4,
           alpha=1,
           size=1,
           log=False,
           pad=1.08,
           w_pad=None,
           h_pad=3,
           fig_size=(3, 3),
           fig_ncol=3,
           save_fig=False,
           fig_path=None,
           fig_name='plot_violin.pdf',
           **kwargs):
    """Violin plot
    """

    if fig_size is None:
        fig_size = mpl.rcParams['figure.figsize']
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, 'figures')
    if list_obs is None:
        list_obs = []
    if list_var is None:
        list_var = []
    for obs in list_obs:
        if (obs not in adata.obs_keys()):
            raise ValueError(f"could not find {obs} in `adata.obs_keys()`")
    for var in list_var:
        if (var not in adata.var_keys()):
            raise ValueError(f"could not find {var} in `adata.var_keys()`")
    if (len(list_obs) > 0):
        df_plot = adata.obs[list_obs].copy()
        if (log):
            df_plot = pd.DataFrame(data=np.log1p(df_plot.values),
                                   index=df_plot.index,
                                   columns=df_plot.columns)
        fig_nrow = int(np.ceil(len(list_obs)/fig_ncol))
        fig = plt.figure(figsize=(fig_size[0]*fig_ncol*1.05,
                         fig_size[1]*fig_nrow))
        for i, obs in enumerate(list_obs):
            ax_i = fig.add_subplot(fig_nrow, fig_ncol, i+1)
            sns.violinplot(ax=ax_i,
                           y=obs,
                           data=df_plot,
                           inner=None,
                           **kwargs)
            sns.stripplot(ax=ax_i,
                          y=obs,
                          data=df_plot,
                          color='black',
                          jitter=jitter,
                          alpha=alpha,
                          s=size)

            ax_i.set_title(obs)
            ax_i.set_ylabel('')
            ax_i.locator_params(axis='y', nbins=6)
            ax_i.tick_params(axis="y", pad=-2)
            ax_i.spines['right'].set_visible(False)
            ax_i.spines['top'].set_visible(False)
        plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
        if (save_fig):
            if (not os.path.exists(fig_path)):
                os.makedirs(fig_path)
            plt.savefig(os.path.join(fig_path, fig_name),
                        pad_inches=1,
                        bbox_inches='tight')
            plt.close(fig)
    if (len(list_var) > 0):
        df_plot = adata.var[list_var].copy()
        if (log):
            df_plot = pd.DataFrame(data=np.log1p(df_plot.values),
                                   index=df_plot.index,
                                   columns=df_plot.columns)
        fig_nrow = int(np.ceil(len(list_obs)/fig_ncol))
        fig = plt.figure(figsize=(fig_size[0]*fig_ncol*1.05,
                                  fig_size[1]*fig_nrow))
        for i, var in enumerate(list_var):
            ax_i = fig.add_subplot(fig_nrow, fig_ncol, i+1)
            sns.violinplot(ax=ax_i,
                           y=var,
                           data=df_plot,
                           inner=None,
                           **kwargs)
            sns.stripplot(ax=ax_i,
                          y=var,
                          data=df_plot,
                          color='black',
                          jitter=jitter,
                          s=size)

            ax_i.set_title(var)
            ax_i.set_ylabel('')
            ax_i.locator_params(axis='y', nbins=6)
            ax_i.tick_params(axis="y", pad=-2)
            ax_i.spines['right'].set_visible(False)
            ax_i.spines['top'].set_visible(False)
        plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
        if (save_fig):
            if (not os.path.exists(fig_path)):
                os.makedirs(fig_path)
            plt.savefig(os.path.join(fig_path, fig_name),
                        pad_inches=1,
                        bbox_inches='tight')
            plt.close(fig)


def hist(adata,
         list_obs=None,
         list_var=None,
         kde=True,
         size=1,
         log=False,
         pad=1.08,
         w_pad=None,
         h_pad=3,
         fig_size=(3, 3),
         fig_ncol=3,
         save_fig=False,
         fig_path=None,
         fig_name='plot_violin.pdf',
         **kwargs
         ):
    """histogram plot
    """

    if fig_size is None:
        fig_size = mpl.rcParams['figure.figsize']
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, 'figures')
    if list_obs is None:
        list_obs = []
    if list_var is None:
        list_var = []
    for obs in list_obs:
        if (obs not in adata.obs_keys()):
            raise ValueError(f"could not find {obs} in `adata.obs_keys()`")
    for var in list_var:
        if (var not in adata.var_keys()):
            raise ValueError(f"could not find {var} in `adata.var_keys()`")

    if (len(list_obs) > 0):
        df_plot = adata.obs[list_obs].copy()
        if (log):
            df_plot = pd.DataFrame(data=np.log1p(df_plot.values),
                                   index=df_plot.index,
                                   columns=df_plot.columns)
        fig_nrow = int(np.ceil(len(list_obs)/fig_ncol))
        fig = plt.figure(figsize=(fig_size[0]*fig_ncol*1.05,
                                  fig_size[1]*fig_nrow))
        for i, obs in enumerate(list_obs):
            ax_i = fig.add_subplot(fig_nrow, fig_ncol, i+1)
            sns.histplot(ax=ax_i,
                         x=obs,
                         data=df_plot,
                         kde=kde,
                         **kwargs)
            ax_i.locator_params(axis='y', nbins=6)
            ax_i.tick_params(axis="y", pad=-2)
            ax_i.spines['right'].set_visible(False)
            ax_i.spines['top'].set_visible(False)
        plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
        if (save_fig):
            if (not os.path.exists(fig_path)):
                os.makedirs(fig_path)
            plt.savefig(os.path.join(fig_path, fig_name),
                        pad_inches=1,
                        bbox_inches='tight')
            plt.close(fig)
    if (len(list_var) > 0):
        df_plot = adata.var[list_var].copy()
        if (log):
            df_plot = pd.DataFrame(data=np.log1p(df_plot.values),
                                   index=df_plot.index,
                                   columns=df_plot.columns)
        fig_nrow = int(np.ceil(len(list_obs)/fig_ncol))
        fig = plt.figure(figsize=(fig_size[0]*fig_ncol*1.05,
                                  fig_size[1]*fig_nrow))
        for i, var in enumerate(list_var):
            ax_i = fig.add_subplot(fig_nrow, fig_ncol, i+1)
            sns.histplot(ax=ax_i,
                         x=var,
                         data=df_plot,
                         kde=kde,
                         **kwargs)
            ax_i.locator_params(axis='y', nbins=6)
            ax_i.tick_params(axis="y", pad=-2)
            ax_i.spines['right'].set_visible(False)
            ax_i.spines['top'].set_visible(False)
        plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
        if (save_fig):
            if (not os.path.exists(fig_path)):
                os.makedirs(fig_path)
            plt.savefig(os.path.join(fig_path, fig_name),
                        pad_inches=1,
                        bbox_inches='tight')
            plt.close(fig)


def pca_variance_ratio(adata,
                       log=True,
                       show_cutoff=True,
                       fig_size=(4, 4),
                       save_fig=None,
                       fig_path=None,
                       fig_name='qc.pdf',
                       pad=1.08,
                       w_pad=None,
                       h_pad=None):
    """Plot the variance ratio.
    """

    if fig_size is None:
        fig_size = mpl.rcParams['figure.figsize']
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, 'figures')

    n_components = len(adata.uns['pca']['variance_ratio'])

    fig = plt.figure(figsize=fig_size)
    if (log):
        plt.plot(range(n_components),
                 np.log(adata.uns['pca']['variance_ratio']))
    else:
        plt.plot(range(n_components),
                 adata.uns['pca']['variance_ratio'])
    if (show_cutoff):
        n_pcs = adata.uns['pca']['n_pcs']
        print(f'the number of selected PC is: {n_pcs}')
        plt.axvline(n_pcs, ls='--', c='red')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Ratio')
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    if (save_fig):
        if (not os.path.exists(fig_path)):
            os.makedirs(fig_path)
        plt.savefig(os.path.join(fig_path, fig_name),
                    pad_inches=1,
                    bbox_inches='tight')
        plt.close(fig)


def pcs_features(adata,
                 log=False,
                 size=3,
                 show_cutoff=True,
                 fig_size=(3, 3),
                 fig_ncol=3,
                 save_fig=None,
                 fig_path=None,
                 fig_name='qc.pdf',
                 pad=1.08,
                 w_pad=None,
                 h_pad=None):
    """Plot features that contribute to the top PCs.
    """

    if fig_size is None:
        fig_size = mpl.rcParams['figure.figsize']
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, 'figures')

    n_pcs = adata.uns['pca']['n_pcs']
    n_features = adata.uns['pca']['PCs'].shape[0]

    fig_nrow = int(np.ceil(n_pcs/fig_ncol))
    fig = plt.figure(figsize=(fig_size[0]*fig_ncol*1.05, fig_size[1]*fig_nrow))

    for i in range(n_pcs):
        ax_i = fig.add_subplot(fig_nrow, fig_ncol, i+1)
        if (log):
            ax_i.scatter(range(n_features),
                         np.log(np.sort(
                             np.abs(adata.uns['pca']['PCs'][:, i],))[::-1]),
                         s=size)
        else:
            ax_i.scatter(range(n_features),
                         np.sort(
                             np.abs(adata.uns['pca']['PCs'][:, i],))[::-1],
                         s=size)
        n_ft_selected_i = len(adata.uns['pca']['features'][f'pc_{i}'])
        if (show_cutoff):
            ax_i.axvline(n_ft_selected_i, ls='--', c='red')
        ax_i.set_xlabel('Feautures')
        ax_i.set_ylabel('Loadings')
        ax_i.locator_params(axis='x', nbins=3)
        ax_i.locator_params(axis='y', nbins=5)
        ax_i.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax_i.set_title(f'PC {i}')
    plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    if (save_fig):
        if (not os.path.exists(fig_path)):
            os.makedirs(fig_path)
        plt.savefig(os.path.join(fig_path, fig_name),
                    pad_inches=1,
                    bbox_inches='tight')
        plt.close(fig)


def variable_genes(adata,
                   show_texts=False,
                   n_texts=10,
                   size=8,
                   text_size=10,
                   fig_size=(4, 4),
                   save_fig=None,
                   fig_path=None,
                   fig_name='plot_variable_genes.pdf',
                   pad=1.08,
                   w_pad=None,
                   h_pad=None):
    """Plot highly variable genes.
    """

    if fig_size is None:
        fig_size = mpl.rcParams['figure.figsize']
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, 'figures')

    means = adata.var['means']
    variances_norm = adata.var['variances_norm']
    mask = adata.var['highly_variable']
    genes = adata.var_names

    fig, ax = plt.subplots(figsize=fig_size)
    ax.scatter(means[~mask],
               variances_norm[~mask],
               s=size,
               c='#1F2433')
    ax.scatter(means[mask],
               variances_norm[mask],
               s=size,
               c='#ce3746')
    ax.set_xscale(value='log')

    if show_texts:
        ids = variances_norm.values.argsort()[-n_texts:][::-1]
        texts = [plt.text(means[i], variances_norm[i], genes[i],
                 fontdict={'family': 'serif',
                           'color': 'black',
                           'weight': 'normal',
                           'size': text_size})
                 for i in ids]
        adjust_text(texts,
                    arrowprops=dict(arrowstyle='-', color='black'))

    ax.set_xlabel('average expression')
    ax.set_ylabel('standardized variance')
    ax.locator_params(axis='x', tight=True)
    ax.locator_params(axis='y', tight=True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    if (save_fig):
        if (not os.path.exists(fig_path)):
            os.makedirs(fig_path)
        fig.savefig(os.path.join(fig_path, fig_name),
                    pad_inches=1,
                    bbox_inches='tight')
        plt.close(fig)


def _scatterplot2d(df,
                   x,
                   y,
                   df_bg=None,
                   list_hue=None,
                   hue_palette=None,
                   drawing_order='sorted',
                   dict_drawing_order=None,
                   size=8,
                   show_texts=False,
                   texts=None,
                   text_size=10,
                   show_contour=False,
                   contour_levels=5,
                   show_bg=False,
                   bg_size=5,
                   bg_alpha=0.2,
                   bg_color='gray',
                   fig_size=None,
                   fig_ncol=3,
                   fig_legend_ncol=1,
                   fig_legend_order=None,
                   vmin=None,
                   vmax=None,
                   alpha=0.8,
                   pad=1.08,
                   w_pad=None,
                   h_pad=None,
                   save_fig=None,
                   fig_path=None,
                   fig_name='scatterplot2d.pdf',
                   copy=False,
                   **kwargs):
    """2d scatter plot

    Parameters
    ----------
    data: `pd.DataFrame`
        Input data structure of shape (n_samples, n_features).
    x: `str`
        Variable in `data` that specify positions on the x axis.
    y: `str`
        Variable in `data` that specify positions on the x axis.
    list_hue: `str`, optional (default: None)
        A list of variables that will produce points with different colors.
    drawing_order: `str` (default: 'sorted')
        The order in which values are plotted, This can be
        one of the following values
        - 'original': plot points in the same order as in input dataframe
        - 'sorted' : plot points with higher values on top.
        - 'random' : plot points in a random order
    fig_size: `tuple`, optional (default: None)
        figure size.
    fig_ncol: `int`, optional (default: 3)
        the number of columns of the figure panel
    fig_legend_order: `dict`,optional (default: None)
        Specified order for the appearance of the annotation keys.
        Only valid for categorical/string variable
        e.g. fig_legend_order = {'ann1':['a','b','c'],'ann2':['aa','bb','cc']}
    fig_legend_ncol: `int`, optional (default: 1)
        The number of columns that the legend has.
    vmin,vmax: `float`, optional (default: None)
        The min and max values are used to normalize continuous values.
        If None, the respective min and max of continuous values is used.
    alpha: `float`, optional (default: 0.8)
        0.0 transparent through 1.0 opaque
    pad: `float`, optional (default: 1.08)
        Padding between the figure edge and the edges of subplots,
        as a fraction of the font size.
    h_pad, w_pad: `float`, optional (default: None)
        Padding (height/width) between edges of adjacent subplots,
        as a fraction of the font size. Defaults to pad.
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_path: `str`, optional (default: None)
        If save_fig is True, specify figure path.
    fig_name: `str`, optional (default: 'scatterplot2d.pdf')
        if save_fig is True, specify figure name.
    Returns
    -------
    None
    """

    if fig_size is None:
        fig_size = mpl.rcParams['figure.figsize']
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, 'figures')

    list_ax = list()
    if list_hue is None:
        list_hue = [None]
    else:
        for hue in list_hue:
            if (hue not in df.columns):
                raise ValueError(f"could not find {hue}")
        if hue_palette is None:
            hue_palette = dict()
        assert isinstance(hue_palette, dict), "`hue_palette` must be dict"
        legend_order = {hue: np.unique(df[hue]) for hue in list_hue
                        if (is_string_dtype(df[hue])
                            or is_categorical_dtype(df[hue]))}
        if (fig_legend_order is not None):
            if (not isinstance(fig_legend_order, dict)):
                raise TypeError("`fig_legend_order` must be a dictionary")
            for hue in fig_legend_order.keys():
                if (hue in legend_order.keys()):
                    legend_order[hue] = fig_legend_order[hue]
                else:
                    print(f"{hue} is ignored for ordering legend labels"
                          "due to incorrect name or data type")

    if dict_drawing_order is None:
        dict_drawing_order = dict()
    assert drawing_order in ['sorted', 'random', 'original'],\
        "`drawing_order` must be one of ['original', 'sorted', 'random']"

    if (len(list_hue) < fig_ncol):
        fig_ncol = len(list_hue)
    fig_nrow = int(np.ceil(len(list_hue)/fig_ncol))
    fig = plt.figure(figsize=(fig_size[0]*fig_ncol*1.05, fig_size[1]*fig_nrow))
    for i, hue in enumerate(list_hue):
        ax_i = fig.add_subplot(fig_nrow, fig_ncol, i+1)
        if show_bg:
            if df_bg is not None:
                sns.scatterplot(ax=ax_i,
                                x=x,
                                y=y,
                                color=bg_color,
                                data=df_bg,
                                alpha=bg_alpha,
                                linewidth=0,
                                s=bg_size)
        if hue is None:
            sc_i = sns.scatterplot(ax=ax_i,
                                   x=x,
                                   y=y,
                                   data=df,
                                   alpha=alpha,
                                   linewidth=0,
                                   s=size,
                                   **kwargs)
        else:
            if (is_string_dtype(df[hue]) or is_categorical_dtype(df[hue])):
                if hue in hue_palette.keys():
                    palette = hue_palette[hue]
                else:
                    palette = None
                if hue in dict_drawing_order.keys():
                    param_drawing_order = dict_drawing_order[hue]
                else:
                    param_drawing_order = drawing_order
                if param_drawing_order == 'sorted':
                    df_updated = df.sort_values(by=hue)
                elif param_drawing_order == 'random':
                    df_updated = df.sample(frac=1, random_state=100)
                else:
                    df_updated = df
                sc_i = sns.scatterplot(ax=ax_i,
                                       x=x,
                                       y=y,
                                       hue=hue,
                                       hue_order=legend_order[hue],
                                       data=df_updated,
                                       alpha=alpha,
                                       linewidth=0,
                                       palette=palette,
                                       s=size,
                                       **kwargs)
                ax_i.legend(bbox_to_anchor=(1, 0.5),
                            loc='center left',
                            ncol=fig_legend_ncol,
                            frameon=False,
                            )
            else:
                vmin_i = df[hue].min() if vmin is None else vmin
                vmax_i = df[hue].max() if vmax is None else vmax
                if hue in dict_drawing_order.keys():
                    param_drawing_order = dict_drawing_order[hue]
                else:
                    param_drawing_order = drawing_order
                if param_drawing_order == 'sorted':
                    df_updated = df.sort_values(by=hue)
                elif param_drawing_order == 'random':
                    df_updated = df.sample(frac=1, random_state=100)
                else:
                    df_updated = df
                sc_i = ax_i.scatter(df_updated[x],
                                    df_updated[y],
                                    c=df_updated[hue],
                                    vmin=vmin_i,
                                    vmax=vmax_i,
                                    alpha=alpha,
                                    s=size)
                cbar = plt.colorbar(sc_i,
                                    ax=ax_i,
                                    pad=0.01,
                                    fraction=0.05,
                                    aspect=40)
                cbar.solids.set_edgecolor("face")
                cbar.ax.locator_params(nbins=5)
        if show_texts:
            if texts is not None:
                plt_texts = [plt.text(df[x][t],
                                      df[y][t],
                                      t,
                                      fontdict={'family': 'serif',
                                                'color': 'black',
                                                'weight': 'normal',
                                                'size': text_size})
                             for t in texts]
                adjust_text(plt_texts,
                            arrowprops=dict(arrowstyle='->', color='black'))
        if show_contour:
            sns.kdeplot(ax=ax_i,
                        data=df,
                        x=x,
                        y=y,
                        alpha=0.9,
                        color='black',
                        levels=contour_levels,
                        **kwargs)
        ax_i.set_xlabel(x)
        ax_i.set_ylabel(y)
        ax_i.locator_params(axis='x', nbins=5)
        ax_i.locator_params(axis='y', nbins=5)
        ax_i.tick_params(axis="both", labelbottom=True, labelleft=True)
        ax_i.set_title(hue)
        list_ax.append(ax_i)
    plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    if save_fig:
        if (not os.path.exists(fig_path)):
            os.makedirs(fig_path)
        plt.savefig(os.path.join(fig_path, fig_name),
                    pad_inches=1,
                    bbox_inches='tight')
        plt.close(fig)
    if copy:
        return list_ax


def _scatterplot2d_plotly(df,
                          x,
                          y,
                          list_hue=None,
                          hue_palette=None,
                          drawing_order='sorted',
                          fig_size=None,
                          fig_ncol=3,
                          fig_legend_order=None,
                          alpha=0.8,
                          save_fig=None,
                          fig_path=None,
                          **kwargs):
    """interactive 2d scatter plot by Plotly

    Parameters
    ----------
    data: `pd.DataFrame`
        Input data structure of shape (n_samples, n_features).
    x: `str`
        Variable in `data` that specify positions on the x axis.
    y: `str`
        Variable in `data` that specify positions on the x axis.
    list_hue: `str`, optional (default: None)
        A list of variables that will produce points with different colors.
    drawing_order: `str` (default: 'sorted')
        The order in which values are plotted, This can be
        one of the following values
        - 'original': plot points in the same order as in input dataframe
        - 'sorted' : plot points with higher values on top.
        - 'random' : plot points in a random order
    fig_size: `tuple`, optional (default: None)
        figure size.
    fig_ncol: `int`, optional (default: 3)
        the number of columns of the figure panel
    fig_legend_order: `dict`,optional (default: None)
        Specified order for the appearance of the annotation keys.
        Only valid for categorical/string variable
        e.g. fig_legend_order = {'ann1':['a','b','c'],'ann2':['aa','bb','cc']}
    fig_legend_ncol: `int`, optional (default: 1)
        The number of columns that the legend has.
    vmin,vmax: `float`, optional (default: None)
        The min and max values are used to normalize continuous values.
        If None, the respective min and max of continuous values is used.
    alpha: `float`, optional (default: 0.8)
        0.0 transparent through 1.0 opaque
    pad: `float`, optional (default: 1.08)
        Padding between the figure edge and the edges of subplots,
        as a fraction of the font size.
    h_pad, w_pad: `float`, optional (default: None)
        Padding (height/width) between edges of adjacent subplots,
        as a fraction of the font size. Defaults to pad.
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_path: `str`, optional (default: None)
        If save_fig is True, specify figure path.
    fig_name: `str`, optional (default: 'scatterplot2d.pdf')
        if save_fig is True, specify figure name.
    Returns
    -------
    None
    """

    if fig_size is None:
        fig_size = mpl.rcParams['figure.figsize']
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, 'figures')

    for hue in list_hue:
        if (hue not in df.columns):
            raise ValueError(f"could not find {hue} in `df.columns`")
    if hue_palette is None:
        hue_palette = dict()
    assert isinstance(hue_palette, dict), "`hue_palette` must be dict"

    assert drawing_order in ['sorted', 'random', 'original'],\
        "`drawing_order` must be one of ['original', 'sorted', 'random']"

    legend_order = {hue: np.unique(df[hue]) for hue in list_hue
                    if (is_string_dtype(df[hue])
                        or is_categorical_dtype(df[hue]))}
    if (fig_legend_order is not None):
        if (not isinstance(fig_legend_order, dict)):
            raise TypeError("`fig_legend_order` must be a dictionary")
        for hue in fig_legend_order.keys():
            if (hue in legend_order.keys()):
                legend_order[hue] = fig_legend_order[hue]
            else:
                print(f"{hue} is ignored for ordering legend labels"
                      "due to incorrect name or data type")

    if (len(list_hue) < fig_ncol):
        fig_ncol = len(list_hue)
    fig_nrow = int(np.ceil(len(list_hue)/fig_ncol))
    fig = plt.figure(figsize=(fig_size[0]*fig_ncol*1.05, fig_size[1]*fig_nrow))
    for hue in list_hue:
        if hue in hue_palette.keys():
            palette = hue_palette[hue]
        else:
            palette = None
        if drawing_order == 'sorted':
            df_updated = df.sort_values(by=hue)
        elif drawing_order == 'random':
            df_updated = df.sample(frac=1, random_state=100)
        else:
            df_updated = df
        fig = px.scatter(df_updated,
                         x=x,
                         y=y,
                         color=hue,
                         opacity=alpha,
                         color_continuous_scale=px.colors.sequential.Viridis,
                         color_discrete_map=palette,
                         **kwargs)
        fig.update_layout(legend={'itemsizing': 'constant'},
                          width=500,
                          height=500)
        fig.show(renderer="notebook")


def umap(adata,
         color=None,
         dict_palette=None,
         n_components=None,
         size=8,
         drawing_order='sorted',
         dict_drawing_order=None,
         show_texts=False,
         texts=None,
         text_size=10,
         fig_size=None,
         fig_ncol=3,
         fig_legend_ncol=1,
         fig_legend_order=None,
         vmin=None,
         vmax=None,
         alpha=1,
         pad=1.08,
         w_pad=None,
         h_pad=None,
         save_fig=None,
         fig_path=None,
         fig_name='scatterplot2d.pdf',
         plolty=False,
         copy=False,
         **kwargs):
    """ Plot coordinates in UMAP

    Parameters
    ----------
    adata: `Anndata`
        Annotated data matrix of shape (n_samples, n_features).
    color: `str`, optional (default: None)
        A list of variables that will produce points with different colors.
        e.g. color = ['anno1', 'anno2']
    dict_palette: `dict`,optional (default: None)
        A dictionary of palettes for different variables in `color`.
        Only valid for categorical/string variables
        e.g. dict_palette = {'ann1': {},'ann2': {}}
    drawing_order: `str` (default: 'sorted')
        The order in which values are plotted, This can be
        one of the following values
        - 'original': plot points in the same order as in input dataframe
        - 'sorted' : plot points with higher values on top.
        - 'random' : plot points in a random order
    size: `int` (default: 8)
        Point size.
    fig_size: `tuple`, optional (default: None)
        figure size.
    fig_ncol: `int`, optional (default: 3)
        the number of columns of the figure panel
    fig_legend_order: `dict`,optional (default: None)
        Specified order for the appearance of the annotation keys.
        Only valid for categorical/string variable
        e.g. fig_legend_order = {'ann1':['a','b','c'],'ann2':['aa','bb','cc']}
    fig_legend_ncol: `int`, optional (default: 1)
        The number of columns that the legend has.
    vmin,vmax: `float`, optional (default: None)
        The min and max values are used to normalize continuous values.
        If None, the respective min and max of continuous values is used.
    alpha: `float`, optional (default: 0.8)
        0.0 transparent through 1.0 opaque
    pad: `float`, optional (default: 1.08)
        Padding between the figure edge and the edges of subplots,
        as a fraction of the font size.
    h_pad, w_pad: `float`, optional (default: None)
        Padding (height/width) between edges of adjacent subplots,
        as a fraction of the font size. Defaults to pad.
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_path: `str`, optional (default: None)
        If save_fig is True, specify figure path.
    fig_name: `str`, optional (default: 'scatterplot2d.pdf')
        if save_fig is True, specify figure name.
    Returns
    -------
    None
    """

    if fig_size is None:
        fig_size = mpl.rcParams['figure.figsize']
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, 'figures')

    if (n_components is None):
        n_components = min(3, adata.obsm['X_umap'].shape[1])
    if n_components not in [2, 3]:
        raise ValueError("n_components should be 2 or 3")
    if (n_components > adata.obsm['X_umap'].shape[1]):
        print(f"`n_components` is greater than the available dimension.\n"
              f"It is corrected to {adata.obsm['X_umap'].shape[1]}")
        n_components = adata.obsm['X_umap'].shape[1]

    if dict_palette is None:
        dict_palette = dict()
    df_plot = pd.DataFrame(index=adata.obs.index,
                           data=adata.obsm['X_umap'],
                           columns=['UMAP'+str(x+1) for x in
                                    range(adata.obsm['X_umap'].shape[1])])
    if color is None:
        list_ax = _scatterplot2d(df_plot,
                                 x='UMAP1',
                                 y='UMAP2',
                                 drawing_order=drawing_order,
                                 size=size,
                                 show_texts=show_texts,
                                 text_size=text_size,
                                 texts=texts,
                                 fig_size=fig_size,
                                 alpha=alpha,
                                 pad=pad,
                                 w_pad=w_pad,
                                 h_pad=h_pad,
                                 save_fig=save_fig,
                                 fig_path=fig_path,
                                 fig_name=fig_name,
                                 copy=copy,
                                 **kwargs)
    else:
        color = list(dict.fromkeys(color))  # remove duplicate keys
        for ann in color:
            if (ann in adata.obs_keys()):
                df_plot[ann] = adata.obs[ann]
                if (not is_numeric_dtype(df_plot[ann])):
                    if 'color' not in adata.uns_keys():
                        adata.uns['color'] = dict()

                    if ann not in dict_palette.keys():
                        if (ann+'_color' in adata.uns['color'].keys()) \
                            and \
                            (all(np.isin(np.unique(df_plot[ann]),
                                         list(adata.uns['color']
                                         [ann+'_color'].keys())))):
                            dict_palette[ann] = \
                                adata.uns['color'][ann+'_color']
                        else:
                            dict_palette[ann] = \
                                generate_palette(adata.obs[ann])
                            adata.uns['color'][ann+'_color'] = \
                                dict_palette[ann].copy()
                    else:
                        if ann+'_color' not in adata.uns['color'].keys():
                            adata.uns['color'][ann+'_color'] = \
                                dict_palette[ann].copy()

            elif (ann in adata.var_names):
                df_plot[ann] = adata.obs_vector(ann)
            else:
                raise ValueError(f"could not find {ann} in `adata.obs.columns`"
                                 " and `adata.var_names`")
        if plolty:
            _scatterplot2d_plotly(df_plot,
                                  x='UMAP1',
                                  y='UMAP2',
                                  list_hue=color,
                                  hue_palette=dict_palette,
                                  drawing_order=drawing_order,
                                  fig_size=fig_size,
                                  fig_ncol=fig_ncol,
                                  fig_legend_order=fig_legend_order,
                                  alpha=alpha,
                                  save_fig=save_fig,
                                  fig_path=fig_path,
                                  **kwargs)
        else:
            list_ax = _scatterplot2d(df_plot,
                                     x='UMAP1',
                                     y='UMAP2',
                                     list_hue=color,
                                     hue_palette=dict_palette,
                                     drawing_order=drawing_order,
                                     dict_drawing_order=dict_drawing_order,
                                     size=size,
                                     show_texts=show_texts,
                                     text_size=text_size,
                                     texts=texts,
                                     fig_size=fig_size,
                                     fig_ncol=fig_ncol,
                                     fig_legend_ncol=fig_legend_ncol,
                                     fig_legend_order=fig_legend_order,
                                     vmin=vmin,
                                     vmax=vmax,
                                     alpha=alpha,
                                     pad=pad,
                                     w_pad=w_pad,
                                     h_pad=h_pad,
                                     save_fig=save_fig,
                                     fig_path=fig_path,
                                     fig_name=fig_name,
                                     copy=copy,
                                     **kwargs)
    if copy:
        return list_ax


def _scatterplot2d_clone(adata,
                         group,
                         target='clone',
                         obsm=None,
                         layer=None,
                         comp1=0,
                         comp2=1,
                         size=8,
                         show_contour=True,
                         fig_size=(5, 5),
                         fig_ncol=3,
                         alpha=0.8,
                         pad=1.08,
                         w_pad=None,
                         h_pad=None,
                         save_fig=None,
                         fig_path=None,
                         fig_name='clone_scatter.pdf',
                         **kwargs,
                         ):
    """ Scatter plot for clone/clone_traj clusters
    """
    if fig_size is None:
        fig_size = mpl.rcParams['figure.figsize']
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, 'figures')

    if (sum(list(map(lambda x: x is not None,
                     [layer, obsm]))) == 2):
        raise ValueError("Only one of `layer` and `obsm` can be used")
    elif obsm is not None:
        if obsm in adata.obsm_keys():
            mat_coord = adata.obsm[obsm]
        else:
            raise ValueError(
                f'could not find {obsm} in `adata.obsm_keys()`')
    elif layer is not None:
        if layer in adata.layers.keys():
            mat_coord = adata.layers[layer]
        else:
            raise ValueError(
                f'could not find {layer} in `adata.layers.keys()`')
    else:
        mat_coord = adata.X

    n_components = 2
    df_X_clone = pd.DataFrame(index=adata.obs.index,
                              data=adata.obsm[f'X_{target}'].A,
                              columns=adata.uns[target]['anno'].index)
    df_clones_clusters = adata.uns[target]['anno'][group].copy()
    df_plot = pd.DataFrame(index=adata.obs.index,
                           data=mat_coord[:, :n_components],
                           columns=[f'Dim{comp1+1}', f'Dim{comp2+1}'])
    for x in np.unique(df_clones_clusters):
        df_plot[f'{target}_cluster_{x}'] = False
        mask = (df_X_clone.loc[:, df_clones_clusters == x].sum(axis=1) > 0)
        df_plot.loc[mask, f'{target}_cluster_{x}'] = True

    dict_palette = generate_palette(np.unique(df_clones_clusters))
    fig_nrow = int(np.ceil(len(np.unique(df_clones_clusters))/fig_ncol))
    fig = plt.figure(figsize=(
        fig_size[0]*fig_ncol,
        fig_size[1]*fig_nrow))

    for i, cl in enumerate(np.unique(df_clones_clusters)):
        ax_i = fig.add_subplot(fig_nrow, fig_ncol, i+1)
        sns.scatterplot(ax=ax_i,
                        x=f'Dim{comp1+1}',
                        y=f'Dim{comp2+1}',
                        color='gray',
                        data=df_plot,
                        alpha=0.2,
                        linewidth=0,
                        s=size)
        sns.scatterplot(ax=ax_i,
                        x=f'Dim{comp1+1}',
                        y=f'Dim{comp2+1}',
                        color=dict_palette[cl],
                        data=df_plot[df_plot[f'{target}_cluster_{cl}']],
                        alpha=alpha,
                        s=size,
                        linewidth=0)
        if show_contour:
            sns.kdeplot(ax=ax_i,
                        data=df_plot[df_plot[f'{target}_cluster_{cl}']],
                        x=f'Dim{comp1+1}',
                        y=f'Dim{comp2+1}',
                        alpha=0.9,
                        color='black',
                        **kwargs)
        ax_i.set_title(f'{target} cluster {cl}')
    plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    if (save_fig):
        if (not os.path.exists(fig_path)):
            os.makedirs(fig_path)
        fig.savefig(os.path.join(fig_path, fig_name),
                    pad_inches=1,
                    bbox_inches='tight')
        plt.close(fig)


def clone_clusters(
    adata,
    group,
    obsm=None,
    layer=None,
    comp1=0,
    comp2=1,
    show_contour=True,
    fig_size=(4, 4),
    fig_ncol=3,
    alpha=0.8,
    pad=1.08,
    w_pad=None,
    h_pad=None,
    save_fig=None,
    fig_path=None,
    fig_name='clone_scatter.pdf',
    **kwargs,
):
    """ Scatter plot for clone clusters
    """
    _scatterplot2d_clone(adata,
                         group,
                         target='clone',
                         obsm=obsm,
                         layer=layer,
                         comp1=comp1,
                         comp2=comp2,
                         show_contour=show_contour,
                         fig_size=fig_size,
                         fig_ncol=fig_ncol,
                         alpha=alpha,
                         pad=pad,
                         w_pad=w_pad,
                         h_pad=h_pad,
                         save_fig=save_fig,
                         fig_path=fig_path,
                         fig_name=fig_name,
                         **kwargs
                         )


def scatter(adata,
            color=None,
            obsm='X_umap',
            layer=None,
            dict_palette=None,
            size=8,
            drawing_order='sorted',
            dict_drawing_order=None,
            show_texts=False,
            texts=None,
            text_size=10,
            fig_size=None,
            fig_ncol=3,
            fig_legend_ncol=1,
            fig_legend_order=None,
            vmin=None,
            vmax=None,
            alpha=1,
            pad=1.08,
            w_pad=None,
            h_pad=None,
            save_fig=None,
            fig_path=None,
            fig_name='scatterplot2d.pdf',
            plolty=False,
            copy=False,
            **kwargs):
    """ Plot coordinates in 2D scatterplot

    Parameters
    ----------
    adata: `Anndata`
        Annotated data matrix of shape (n_samples, n_features).
    color: `str`, optional (default: None)
        A list of variables that will produce points with different colors.
        e.g. color = ['anno1', 'anno2']
    dict_palette: `dict`,optional (default: None)
        A dictionary of palettes for different variables in `color`.
        Only valid for categorical/string variables
        e.g. dict_palette = {'ann1': {},'ann2': {}}
    drawing_order: `str` (default: 'sorted')
        The order in which values are plotted, This can be
        one of the following values
        - 'original': plot points in the same order as in input dataframe
        - 'sorted' : plot points with higher values on top.
        - 'random' : plot points in a random order
    size: `int` (default: 8)
        Point size.
    fig_size: `tuple`, optional (default: None)
        figure size.
    fig_ncol: `int`, optional (default: 3)
        the number of columns of the figure panel
    fig_legend_order: `dict`,optional (default: None)
        Specified order for the appearance of the annotation keys.
        Only valid for categorical/string variable
        e.g. fig_legend_order = {'ann1':['a','b','c'],'ann2':['aa','bb','cc']}
    fig_legend_ncol: `int`, optional (default: 1)
        The number of columns that the legend has.
    vmin,vmax: `float`, optional (default: None)
        The min and max values are used to normalize continuous values.
        If None, the respective min and max of continuous values is used.
    alpha: `float`, optional (default: 0.8)
        0.0 transparent through 1.0 opaque
    pad: `float`, optional (default: 1.08)
        Padding between the figure edge and the edges of subplots,
        as a fraction of the font size.
    h_pad, w_pad: `float`, optional (default: None)
        Padding (height/width) between edges of adjacent subplots,
        as a fraction of the font size. Defaults to pad.
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_path: `str`, optional (default: None)
        If save_fig is True, specify figure path.
    fig_name: `str`, optional (default: 'scatterplot2d.pdf')
        if save_fig is True, specify figure name.
    Returns
    -------
    None
    """

    if fig_size is None:
        fig_size = mpl.rcParams['figure.figsize']
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, 'figures')

    if (sum(list(map(lambda x: x is not None,
                     [layer, obsm]))) == 2):
        raise ValueError("Only one of `layer` and `obsm` can be used")
    elif obsm is not None:
        if obsm in adata.obsm_keys():
            mat_coord = adata.obsm[obsm]
        else:
            raise ValueError(
                f'could not find {obsm} in `adata.obsm_keys()`')
    elif layer is not None:
        if layer in adata.layers.keys():
            mat_coord = adata.layers[layer]
        else:
            raise ValueError(
                f'could not find {layer} in `adata.layers.keys()`')
    else:
        mat_coord = adata.X

    if dict_palette is None:
        dict_palette = dict()
    df_plot = pd.DataFrame(index=adata.obs.index,
                           data=mat_coord[:, :2],
                           columns=['Dim1', 'Dim2'])
    if color is None:
        list_ax = _scatterplot2d(df_plot,
                                 x='Dim1',
                                 y='Dim2',
                                 drawing_order=drawing_order,
                                 size=size,
                                 show_texts=show_texts,
                                 text_size=text_size,
                                 texts=texts,
                                 fig_size=fig_size,
                                 alpha=alpha,
                                 pad=pad,
                                 w_pad=w_pad,
                                 h_pad=h_pad,
                                 save_fig=save_fig,
                                 fig_path=fig_path,
                                 fig_name=fig_name,
                                 copy=copy,
                                 **kwargs)
    else:
        color = list(dict.fromkeys(color))  # remove duplicate keys
        for ann in color:
            if (ann in adata.obs_keys()):
                df_plot[ann] = adata.obs[ann]
                if (not is_numeric_dtype(df_plot[ann])):
                    if 'color' not in adata.uns_keys():
                        adata.uns['color'] = dict()

                    if ann not in dict_palette.keys():
                        if (ann+'_color' in adata.uns['color'].keys()) \
                            and \
                            (all(np.isin(np.unique(df_plot[ann]),
                                         list(adata.uns['color']
                                         [ann+'_color'].keys())))):
                            dict_palette[ann] = \
                                adata.uns['color'][ann+'_color']
                        else:
                            dict_palette[ann] = \
                                generate_palette(adata.obs[ann])
                            adata.uns['color'][ann+'_color'] = \
                                dict_palette[ann].copy()
                    else:
                        if ann+'_color' not in adata.uns['color'].keys():
                            adata.uns['color'][ann+'_color'] = \
                                dict_palette[ann].copy()

            elif (ann in adata.var_names):
                df_plot[ann] = adata.obs_vector(ann)
            else:
                raise ValueError(f"could not find {ann} in `adata.obs.columns`"
                                 " and `adata.var_names`")
        if plolty:
            _scatterplot2d_plotly(df_plot,
                                  x='Dim1',
                                  y='Dim2',
                                  list_hue=color,
                                  hue_palette=dict_palette,
                                  drawing_order=drawing_order,
                                  fig_size=fig_size,
                                  fig_ncol=fig_ncol,
                                  fig_legend_order=fig_legend_order,
                                  alpha=alpha,
                                  save_fig=save_fig,
                                  fig_path=fig_path,
                                  **kwargs)
        else:
            list_ax = _scatterplot2d(df_plot,
                                     x='Dim1',
                                     y='Dim2',
                                     list_hue=color,
                                     hue_palette=dict_palette,
                                     drawing_order=drawing_order,
                                     dict_drawing_order=dict_drawing_order,
                                     size=size,
                                     show_texts=show_texts,
                                     text_size=text_size,
                                     texts=texts,
                                     fig_size=fig_size,
                                     fig_ncol=fig_ncol,
                                     fig_legend_ncol=fig_legend_ncol,
                                     fig_legend_order=fig_legend_order,
                                     vmin=vmin,
                                     vmax=vmax,
                                     alpha=alpha,
                                     pad=pad,
                                     w_pad=w_pad,
                                     h_pad=h_pad,
                                     save_fig=save_fig,
                                     fig_path=fig_path,
                                     fig_name=fig_name,
                                     copy=copy,
                                     **kwargs)
    if copy:
        return list_ax


def scatter_3d(adata,
               color=None,
               obsm='X_umap',
               layer=None,
               dict_palette=None,
               comp1=0,
               comp2=1,
               comp3=2,
               alpha=1,
               width=500,
               height=500,
               **kwargs):

    if (sum(list(map(lambda x: x is not None,
                     [layer, obsm]))) == 2):
        raise ValueError("Only one of `layer` and `obsm` can be used")
    elif obsm is not None:
        if obsm in adata.obsm_keys():
            mat_coord = adata.obsm[obsm]
        else:
            raise ValueError(
                f'could not find {obsm} in `adata.obsm_keys()`')
    elif layer is not None:
        if layer in adata.layers.keys():
            mat_coord = adata.layers[layer]
        else:
            raise ValueError(
                f'could not find {layer} in `adata.layers.keys()`')
    else:
        mat_coord = adata.X

    assert mat_coord.shape[1] >= 3, "At least 3 componets are required"

    if dict_palette is None:
        dict_palette = dict()
    if color is None:
        color = []
        return "No `color` is specified"
    else:
        color = list(dict.fromkeys(color))  # remove duplicate keys
        df_plot = pd.DataFrame(index=adata.obs.index,
                               data=mat_coord,
                               columns=['Dim'+str(x+1) for x in
                                        range(mat_coord.shape[1])])
        for ann in color:
            if (ann in adata.obs_keys()):
                df_plot[ann] = adata.obs[ann]
                if (not is_numeric_dtype(df_plot[ann])):
                    if 'color' not in adata.uns_keys():
                        adata.uns['color'] = dict()

                    if ann not in dict_palette.keys():
                        if (ann+'_color' in adata.uns['color'].keys()) \
                            and \
                            (all(np.isin(np.unique(df_plot[ann]),
                                         list(adata.uns['color'].keys())))):
                            dict_palette[ann] = \
                                adata.uns['color'][ann+'_color']
                        else:
                            dict_palette[ann] = \
                                generate_palette(adata.obs[ann])
                            adata.uns['color'][ann+'_color'] = \
                                dict_palette[ann].copy()
                    else:
                        if ann+'_color' not in adata.uns['color'].keys():
                            adata.uns['color'][ann+'_color'] = \
                                dict_palette[ann].copy()

            elif (ann in adata.var_names):
                df_plot[ann] = adata.obs_vector(ann)
            else:
                raise ValueError(f"could not find {ann} in `adata.obs.columns`"
                                 " and `adata.var_names`")

        for ann in color:
            fig = px.scatter_3d(
                df_plot,
                x='Dim'+str(comp1+1),
                y='Dim'+str(comp2+1),
                z='Dim'+str(comp3+1),
                color=ann,
                opacity=alpha,
                color_continuous_scale=px.colors.sequential.Viridis,
                color_discrete_map=adata.uns['color'][ann+'_color']
                if ann+'_color' in adata.uns['color'].keys() else {},
                **kwargs
            )
            fig.update_traces(marker=dict(size=2))
            fig.update_layout(legend={'itemsizing': 'constant'},
                              width=width,
                              height=height,
                              scene=dict(aspectmode='cube'))
            fig.show(renderer="notebook")


def clones(
    adata,
    ids=None,
    color=None,
    obsm=None,
    layer=None,
    comp1=0,
    comp2=1,
    dict_palette=None,
    size=50,
    bg_size=5,
    drawing_order='sorted',
    dict_drawing_order=None,
    show_texts=False,
    texts=None,
    text_size=10,
    show_contour=False,
    contour_levels=5,
    fig_size=None,
    fig_ncol=3,
    fig_legend_ncol=1,
    fig_legend_order=None,
    vmin=None,
    vmax=None,
    alpha=1,
    pad=1.08,
    bg_alpha=0.2,
    bg_color='gray',
    w_pad=None,
    h_pad=None,
    save_fig=None,
    fig_path=None,
    fig_name='clone.pdf',
    copy=False,
    **kwargs,
):
    """ Scatter plot for specified clones
    """
    if fig_size is None:
        fig_size = mpl.rcParams['figure.figsize']
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, 'figures')
    if dict_palette is None:
        dict_palette = dict()

    if (sum(list(map(lambda x: x is not None,
                     [layer, obsm]))) == 2):
        raise ValueError("Only one of `layer` and `obsm` can be used")
    elif obsm is not None:
        if obsm in adata.obsm_keys():
            mat_coord = adata.obsm[obsm]
        else:
            raise ValueError(
                f'could not find {obsm} in `adata.obsm_keys()`')
    elif layer is not None:
        if layer in adata.layers.keys():
            mat_coord = adata.layers[layer]
        else:
            raise ValueError(
                f'could not find {layer} in `adata.layers.keys()`')
    else:
        mat_coord = adata.X.A

    ind_clones = np.where(
        np.isin(adata.uns['clone']['anno'].index, ids))[0]
    ind_cells = np.where(
        adata.obsm['X_clone'][:, ind_clones].sum(axis=1) > 0)[0]

    df_plot = pd.DataFrame(index=adata.obs.index,
                           data=mat_coord[:, [comp1, comp2]],
                           columns=[f'Dim{comp1+1}', f'Dim{comp2+1}'])
    df_plot['selected'] = False
    df_plot.iloc[ind_cells, 2] = True

    if color is None:
        list_ax = _scatterplot2d(df_plot[df_plot['selected']],
                                 x=f'Dim{comp1+1}',
                                 y=f'Dim{comp2+1}',
                                 show_bg=True,
                                 df_bg=df_plot,
                                 bg_alpha=bg_alpha,
                                 bg_color=bg_color,
                                 bg_size=bg_size,
                                 drawing_order=drawing_order,
                                 size=size,
                                 show_texts=show_texts,
                                 text_size=text_size,
                                 texts=texts,
                                 show_contour=show_contour,
                                 contour_levels=contour_levels,
                                 fig_size=fig_size,
                                 alpha=alpha,
                                 pad=pad,
                                 w_pad=w_pad,
                                 h_pad=h_pad,
                                 save_fig=save_fig,
                                 fig_path=fig_path,
                                 fig_name=fig_name,
                                 copy=copy,
                                 **kwargs)
    else:
        color = list(dict.fromkeys(color))  # remove duplicate keys
        for ann in color:
            if (ann in adata.obs_keys()):
                df_plot[ann] = adata.obs[ann]
                if (not is_numeric_dtype(df_plot[ann])):
                    if 'color' not in adata.uns_keys():
                        adata.uns['color'] = dict()

                    if ann not in dict_palette.keys():
                        if (ann+'_color' in adata.uns['color'].keys()) \
                            and \
                            (all(np.isin(np.unique(df_plot[ann]),
                                         list(adata.uns['color']
                                         [ann+'_color'].keys())))):
                            dict_palette[ann] = \
                                adata.uns['color'][ann+'_color']
                        else:
                            dict_palette[ann] = \
                                generate_palette(adata.obs[ann])
                            adata.uns['color'][ann+'_color'] = \
                                dict_palette[ann].copy()
                    else:
                        if ann+'_color' not in adata.uns['color'].keys():
                            adata.uns['color'][ann+'_color'] = \
                                dict_palette[ann].copy()

            elif (ann in adata.var_names):
                df_plot[ann] = adata.obs_vector(ann)
            else:
                raise ValueError(f"could not find {ann} in `adata.obs.columns`"
                                 " and `adata.var_names`")
        list_ax = _scatterplot2d(df_plot[df_plot['selected']],
                                 x=f'Dim{comp1+1}',
                                 y=f'Dim{comp2+1}',
                                 show_bg=True,
                                 df_bg=df_plot,
                                 bg_alpha=bg_alpha,
                                 bg_color=bg_color,
                                 bg_size=bg_size,
                                 list_hue=color,
                                 hue_palette=dict_palette,
                                 drawing_order=drawing_order,
                                 dict_drawing_order=dict_drawing_order,
                                 size=size,
                                 show_texts=show_texts,
                                 text_size=text_size,
                                 texts=texts,
                                 show_contour=show_contour,
                                 contour_levels=contour_levels,
                                 fig_size=fig_size,
                                 fig_ncol=fig_ncol,
                                 fig_legend_ncol=fig_legend_ncol,
                                 fig_legend_order=fig_legend_order,
                                 vmin=vmin,
                                 vmax=vmax,
                                 alpha=alpha,
                                 pad=pad,
                                 w_pad=w_pad,
                                 h_pad=h_pad,
                                 save_fig=save_fig,
                                 fig_path=fig_path,
                                 fig_name=fig_name,
                                 copy=copy,
                                 **kwargs)
    if copy:
        return list_ax


def clone_dendrogram(
    adata,
    color_threshold=None,
    no_labels=True,
    fig_size=(12, 4),
    save_fig=None,
    fig_path=None,
    fig_name='clone_dendrogram.pdf',
    **kwargs
):
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, 'figures')
    Z = linkage(
        adata.uns['clone']['distance'],
        'ward')
    fig = plt.figure(figsize=fig_size)
    _ = dendrogram(
        Z,
        color_threshold=color_threshold,
        no_labels=no_labels,
        **kwargs)
    if color_threshold is not None:
        plt.axhline(y=color_threshold, c='#7A1A3A')
    if save_fig:
        if (not os.path.exists(fig_path)):
            os.makedirs(fig_path)
        fig.savefig(os.path.join(fig_path, fig_name),
                    pad_inches=1,
                    bbox_inches='tight')
        plt.close(fig)


def cluster_graph(adata,
                  obsm=None,
                  force=False,
                  **kwargs):
    if obsm is None or obsm not in adata.obsm:
        raise ValueError(f'{obsm} not found in adata.obsm')
    if 'cluster_edgelist' in adata.uns and not force:
        print('graph already built, using previous (set force=True to clobber)')
    else:
        print(f"Building k-NN graph based on coordinates in obsm.{obsm}")
        build_graph(adata, obsm=obsm)

    ax = clones(adata, obsm=obsm, copy=True)[0]
    G = nx.from_pandas_edgelist(adata.uns['cluster_edgelist'])
    nx.draw(G, pos=adata.uns['cluster_pos'], with_labels=True,
            font_color='white', ax=ax, **kwargs)
    # plt.show()


def cluster_pie_graph(adata,
                      obsm=None,
                      force=False,
                      grey_out_ambiguous = True):
    if obsm is None or obsm not in adata.obsm:
        raise ValueError(f'{obsm} not found in adata.obsm')
    if 'cluster_edgelist' in adata.uns and not force:
        print('graph already built, using previous (set force=True to clobber)')
    else:
        print(f"Building k-NN graph based on coordinates in obsm.{obsm}")
        build_graph(adata, obsm=obsm)

    G = nx.from_pandas_edgelist(adata.uns['cluster_edgelist'])

    def _draw_pie_marker(xs, ys, ratios, sizes, colors, ax):
        epsilon = 1e-4
        assert sum(ratios) <= 1+epsilon, 'sum of ratios needs to be ~1'

        markers = []
        previous = 0
        # calculate the points of the pie pieces
        for color, ratio in zip(colors, ratios):
            this = 2 * np.pi * ratio + previous
            x = [0] + np.cos(np.linspace(previous, this, 10)).tolist() + [0]
            y = [0] + np.sin(np.linspace(previous, this, 10)).tolist() + [0]
            xy = np.column_stack([x, y])
            previous = this
            markers.append({'marker': xy, 's': np.abs(xy).max()
                           ** 2*np.array(sizes), 'facecolor': color})

        # scatter each of the pie pieces to create pies
        for marker in markers:
            ax.scatter(xs, ys, **marker)

    clone_metaclone = adata.uns['clone']['anno']
    clone_anno = adata.uns['clone']['anno']['hierarchical']
    clone_metaclone_df = pd.get_dummies(clone_anno)
    cell_metaclone_mat = (adata.obsm['X_clone'] @ clone_metaclone_df.values)
    n_metaclones_per_cell = (cell_metaclone_mat > 0).astype(int).sum(axis=1)
    
    # In case of multiple occurrences of the maximum values, 
    # the indices corresponding to the first occurrence are returned.
    adata.obs=adata.obs.assign(metaclone = clone_metaclone_df.columns[cell_metaclone_mat.argmax(axis=1)])
    color_dict = generate_palette(adata.obs['metaclone'])

    if grey_out_ambiguous:
        adata.obs.loc[cell_metaclone_mat.sum(axis=1)==0,'metaclone'] = "0"
        adata.obs.loc[n_metaclones_per_cell>1, 'metaclone'] = "0"
        color_dict["0"]='darkgrey'

    clust_pos = adata.uns['cluster_pos']

    ax = clones(adata, obsm=obsm, copy=True)[0]
    nx.draw_networkx_edges(G, pos=clust_pos, ax=ax)

    for i in range(clust_pos.shape[0]):
        metaclone_counts_by_cluster = adata.obs.value_counts(
            ['cluster', 'metaclone'])
        cts = metaclone_counts_by_cluster[i, :]
        size = cts.sum()
        fracs = cts/cts.sum()
        _draw_pie_marker(clust_pos[i, 0], clust_pos[i, 1], fracs, size, [
                         color_dict[j] for j in cts.index], ax)
    


def metaclone_violin(adata,
           list_var=None,
           list_metaclone=None,
           jitter=0.4,
           alpha=1,
           size=1,
           log=False,
           pad=1.08,
           w_pad=None,
           h_pad=3,
           fig_size=(3, 3),
           fig_ncol=3,
           save_fig=False,
           fig_path=None,
           fig_name='plot_metaclone_violin.pdf',
           **kwargs):
    """Violin plot by metaclone
    """

    if fig_size is None:
        fig_size = mpl.rcParams['figure.figsize']
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, 'figures')
    
    metaclones = sorted(pd.unique(adata.uns['clone']['anno']['hierarchical']))
    if list_metaclone is None:
        list_metaclone = list(metaclones)
    if list_var is None:
        list_var = []

    for meta in list_metaclone:
        if (meta not in metaclones):
            raise ValueError(f"could not find {meta} in `adata.uns['clone']['anno']`")
    for var in list_var:
        if (var not in adata.var_names):
            raise ValueError(f"could not find {var} in `adata.var_names`")
    
    color_dict = generate_palette(pd.unique(adata.uns['clone']['anno']['hierarchical']))
    df_plot = get_vars_by_metaclone(adata, var_subset=list_var, metaclone_subset=list_metaclone)
    if (log):
        df_plot = df_plot.transform(np.log1p)

    if (len(list_var) > 0):
        fig_nrow = int(np.ceil(len(list_var)/fig_ncol))
        fig = plt.figure(figsize=(fig_size[0]*fig_ncol*1.05,
                                  fig_size[1]*fig_nrow))
        for i, var in enumerate(list_var):
            ax_i = fig.add_subplot(fig_nrow, fig_ncol, i+1)
            sns.violinplot(ax=ax_i,
                           x='metaclone',
                           y=var,
                           data=df_plot,
                           inner=None,
                           palette=color_dict,
                           **kwargs)
            sns.stripplot(ax=ax_i,
                          x='metaclone',
                          y=var,
                          data=df_plot,
                          color='black',
                          jitter=jitter,
                          s=size)

            ax_i.set_title(var)
            ax_i.set_ylabel('')
            ax_i.locator_params(axis='y', nbins=6)
            ax_i.tick_params(axis="y", pad=-2)
            ax_i.spines['right'].set_visible(False)
            ax_i.spines['top'].set_visible(False)
        plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
        if (save_fig):
            if (not os.path.exists(fig_path)):
                os.makedirs(fig_path)
            plt.savefig(os.path.join(fig_path, fig_name),
                        pad_inches=1,
                        bbox_inches='tight')
            plt.close(fig)

def metaclone_lineplot(adata,
           list_var=None,
           list_metaclone=None,
           max_pseudotime=inf,
           jitter=0.4,
           alpha=1,
           size=1,
           log=False,
           pad=1.08,
           w_pad=None,
           h_pad=3,
           fig_size=(3, 3),
           fig_ncol=3,
           save_fig=False,
           fig_path=None,
           fig_name='plot_metaclone_lines.pdf',
           **kwargs):
    """Line plot by metaclone over pseudotime
    """
    
    if 'pseudotime' not in adata.obs_keys():
        raise ValueError("'pseudotime' not found in adata.obs_keys()")


    if fig_size is None:
        fig_size = mpl.rcParams['figure.figsize']
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, 'figures')
    
    metaclones = sorted(pd.unique(adata.uns['clone']['anno']['hierarchical']))
    if list_metaclone is None:
        list_metaclone = list(metaclones)
    if list_var is None:
        list_var = []

    for meta in list_metaclone:
        if (meta not in metaclones):
            raise ValueError(f"could not find {meta} in `adata.uns['clone']['anno']`")
    for var in list_var:
        if (var not in adata.var_names):
            raise ValueError(f"could not find {var} in `adata.var_names`")
    
    color_dict = generate_palette(pd.unique(adata.uns['clone']['anno']['hierarchical']))
    df_plot = get_vars_by_metaclone(adata, var_subset=list_var, metaclone_subset=list_metaclone)
    df_plot = df_plot.query(f'pseudotime<={max_pseudotime}')
    # df_plot = df_plot.sample(frac = 1, replace=False) # shuffle
    if (log):
        df_plot = df_plot.transform(np.log1p)

    if (len(list_var) > 0):
        fig_nrow = int(np.ceil(len(list_var)/fig_ncol))
        fig = plt.figure(figsize=(fig_size[0]*fig_ncol*1.05,
                                  fig_size[1]*fig_nrow))
        for i, var in enumerate(list_var):
            ax_i = fig.add_subplot(fig_nrow, fig_ncol, i+1)
            # sns.stripplot(ax=ax_i,
            #     x='pseudotime',
            #     y=var,
            #     hue='metaclone',
            #     data=df_plot,
            #     legend=None,
            #     palette=color_dict,
            #     size=1.5,
            #     # native_scale=True, # requires seaborn>=0.12
            #     **kwargs)
            sns.lineplot(ax=ax_i,
                           x='pseudotime',
                           y=var,
                           hue='metaclone',
                           data=df_plot,
                           legend=None,
                           palette=color_dict,
                           **kwargs)

            ax_i.set_title(var)
            ax_i.set_ylabel('')
            ax_i.locator_params(axis='y', nbins=6)
            ax_i.tick_params(axis="y", pad=-2)
            ax_i.spines['right'].set_visible(False)
            ax_i.spines['top'].set_visible(False)
        
        plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
        if (save_fig):
            if (not os.path.exists(fig_path)):
                os.makedirs(fig_path)
            plt.savefig(os.path.join(fig_path, fig_name),
                        pad_inches=1,
                        bbox_inches='tight')
            plt.close(fig)
