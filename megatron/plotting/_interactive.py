import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster import hierarchy
import ipywidgets as widgets
from ipywidgets import interact_manual
from IPython.display import display
import seaborn as sns
import elpigraph
import contextlib
import os
%matplotlib inline

def make_dendrogram(Z, num_clusters):
    plt.title("Hierarchical Clustering Dendrogram (truncated)")
    plt.xlabel('Clone Id or (Meta-Clone Size)')
    plt.ylabel('Distance')
    hierarchy.dendrogram(
        Z,
        truncate_mode = 'lastp',
        p = num_clusters,
        leaf_rotation = 90.,
        leaf_font_size = 10.,
        show_contracted = True
        # labels = ???
    )
    plt.show()

def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                func(*a, **ka)
    return wrapper



    print("This won't be printed.")

#@supress_stdout
def make_elpi(cl_i):
    # epg_i = elpigraph.computeElasticPrincipalTree(X = cl_i,
    #                                                 NumNodes = 50,
    #                                                 n_cores = 1,
    #                                                 drawAccuracyComplexity = False, drawEnergy = False,drawPCAView = False,
    #                                                 Do_PCA=False,CenterData=False)
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        epg_i = elpigraph.computeElasticPrincipalCurve(X = cl_i,
            NumNodes = 20,
            n_cores = 4,
            #drawAccuracyComplexity = False, 
            #drawEnergy = False,
            #drawPCAView = False,
            Do_PCA=False,CenterData=False,verbose=False,
            alpha=2) #,TrimmingRadius=500)    
    return epg_i

def plot_dendrogram_and_space(Z, df_coords, df_clones, num_clusters=5):
    plt.clf()
    make_dendrogram(Z, num_clusters)

    clone_clusters = fcluster(Z, num_clusters, criterion='maxclust')
    df_clones_clusters = pd.Series(data=clone_clusters,
                                   index=df_clones.columns)
    dict_epg = dict()
    for i,x in enumerate(np.unique(clone_clusters)):
        clone_cluster_i = df_coords.loc[df_clones.index[df_clones[df_clones_clusters.index[df_clones_clusters==x]].sum(axis=1)>0]]
        coord_vals_i = clone_cluster_i[['0','1']].values
        dict_epg[i] = make_elpi(coord_vals_i)
    if num_clusters < 3:
        n_col = num_clusters
    else:
        n_col = 3
    if n_col < 3:
        n_row = 1
    else:
        n_row = int(np.ceil(len(np.unique(clone_clusters))/n_col))
    color_palette = sns.color_palette(n_colors=len(np.unique(clone_clusters)))
    colormap = {i: color_palette[i-1] for i in dict_epg.keys()}
    fig, axs = plt.subplots(n_row,n_col, figsize=(5*n_col, 5*n_row), squeeze=False)

    if num_clusters < 3:
        for i,x in enumerate(np.unique(clone_clusters)):
            clone_cluster_i = df_coords.loc[df_clones.index[df_clones[df_clones_clusters.index[df_clones_clusters==x]].sum(axis=1)>0]]
            axs[0,i].scatter(df_coords['0'], df_coords['1'],
                                                        c='gray',alpha=0.05,edgecolors='none')
            axs[0,i].scatter(clone_cluster_i['0'], clone_cluster_i['1'],
                                                        c=[colormap[i]],edgecolors='none',alpha=0.6)
            axs[0,i].set_title('Meta-Clone '+str(x))

    else:
        for i,x in enumerate(np.unique(clone_clusters)):
            clone_cluster_i = df_coords.loc[df_clones.index[df_clones[df_clones_clusters.index[df_clones_clusters==x]].sum(axis=1)>0]]
            axs[int(np.floor(i/n_col)),i%n_col].scatter(df_coords['0'], df_coords['1'],
                                                        c='gray',alpha=0.05,edgecolors='none')
            axs[int(np.floor(i/n_col)),i%n_col].scatter(clone_cluster_i['0'], clone_cluster_i['1'],
                                                        c=[colormap[i]],edgecolors='none',alpha=0.6)
            axs[int(np.floor(i/n_col)),i%n_col].set_title('Meta-clone '+str(x))


    plt.figure(figsize=(6,6))
    ax = sns.scatterplot(x=df_coords['0'], y=df_coords['1'],linewidth=0,alpha=0.1, color='gray')
    for i in dict_epg.keys():
        epg_i = dict_epg[i]
        for ii in range(epg_i[0]['Edges'][0].shape[0]):
            edge_ii = epg_i[0]['Edges'][0][ii]
            nodes_pos_ii = epg_i[0]['NodePositions'][edge_ii,:]
            plt.plot(nodes_pos_ii[:,0],nodes_pos_ii[:,1],
                 color=colormap[i],
                 linewidth=5,
            )
    plt.title("All Meta-Clones")

def full_viz(adata):
    Z = adata.uns[target]['distance']
    df_clones = adata.obsm[f'X_clone']
    df_coords = adata.X

    interact_manual(
        plot_dendrogram_and_space,
        Z=fixed(Z),
        df_coords=fixed(df_coords),
        num_clusters=widgets.IntSlider(min=1, max=30,
                                      step=1, description="Meta-clones: "),
        df_clones=fixed(df_clones)
    )
