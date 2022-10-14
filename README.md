<h1 align="center"><img src="./docs/source/_static/img/logo_200x204.png?raw=true" width="100px"> MEGATRON (MEGA TRajectories of clONes)<img src="./docs/source/_static/img/logo_200x204.png?raw=true" width="100px"></h1>
<p align="center">
  <a href="https://github.com/pinellolab/MEGATRON/actions/workflows/CI.yml"><img alt="CI Status" src="https://github.com/pinellolab/megatron/actions/workflows/CI.yml/badge.svg"></a>
</p>

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Datasets](#datasets)
- [Distances](#distances)
  * [Centroids](#centroids)
  * [Wasserstein](#wasserstein)
  * [Multi Neighbor Network (MNN)](#multi-neighbor-network-mnn)
  * [Geodesic](#geodesic)
- [Tutorials](#tutorials)

## Introduction

MEGATRON, is a Python package to process and interactively visualize clonal trajectories and based on the idea of metaclones. Briefly metaclones are groups of clones that share similar transcriptomic or epigenomic profiles across time points and developmental trajectories. Based on this grouping we can create consensus trajectories i.e. trajectories that summarize similar clones with shared fates. We have tested this method on  recent lineage tracing technologies (Larry, CellTagging, profiling of mitochrondial mutations in scATAC-seq) that simultaneously track clonal relationships and transcriptional or chromatin accessibility states. 

MEGATRON also enable the dection of important genes or transcription factor binding events associated with each metaclone or diverging between two selected metaclones. Importantly, metaclones can partially overlap in the same embedding space, therefore potential enabling to discover of early events associated with cell fate not detacable by current discrete clusetering analyses.




## Installation

```bash
pip install git+https://github.com/pinellolab/MEGATRON
```


## Datasets

* Larry [download](https://mega.nz/folder/gVhFkYaA#FH3S3VoxxeIoTW6aR-sWcA)

* Celltagging [download](https://mega.nz/folder/EJ4FXIYC#8Kx_qiPl4DTBko3AJBjufQ)

* mtscATAC (upload pending)

## Distances
Megatron provides a variety of algorithms to compute distance between individual clonal trajectories. Each algorithm is designed to consider the dimensions provided and quantify distances between clonal trajectories. The specifics of each approach make some better suited for different data and explain the distribution of computational load each will require. Megatron will provide users with the option to attempt different algorithms to observe which is best suited for their unique data.

In the following sections, we will define how  calculate the distance (D) between two clonal trajectories i and j. The subset of cells within i will be denoted as c_i. The subset of cells within i sequenced at time t will be denoted as c_i,t. The set of all cells within the lineage tracing data will be denoted as C.

### Centroids
As an initial naive solution, we first propose the centroids distance. Under this distance, D(i,j) is determined by identifying the Euclidean center of c_i and measuring the distance between that center and the center identified from the embedding of c_j. 

### Wasserstein
The Wasserstein distance relies on the 1-dimensional earth mover's distance (denoted as W). This distance is defined as the minimum amount of 'work' required to transform one 1-dimensional distribution into the other, where 'work' is measured as the amount of distribution weight to be moved multiplied by the distance between distributions. We reasoned that clonal trajectories that follow similar paths over time should require less work to transform their distributions into each other.

For every dimension, this distance calculates W(c_i,t, c_j,t). Then, these values are averaged over all timepoints t and all dimensions. The resulting average is D(i,j). Thus, clonal trajectories whose cells at a given timepoint are embedded adjacent to one another require less work to transform into each other and have smaller distances between them.

### Multi Neighbor Network (MNN)
For all cells, the MNN distance first computes either a K-nearest neighbors graph or a radius graph from the embedding data (the parameters for K or exact radius should be provided by the user, although we provide sensible defaults). This neighbors network is then queried for all c_i and c_j. The set of neighbors of c_i is denoted as cn_i. We reasoned that clonal trajectories that follow similar paths over time should have neighbors belonging to the other clone.

We first calculate the fraction of cn_i,t belonging to c_j,t and the fraction of cn_j,t belonging to c_i,t. The reciprocal of these values is averaged to provide D(i,j).

### Geodesic
Geodesic distance is a graph-based method for calculating the distance between clones. It can leverage the temporal information (if such information is not available, cells are considered to be of the same timepoint). Geodesic distance consists of four major steps: 
- building a connected k-nearest neighbor (KNN) graph. Briefly, cells are first over-clustered. Then an initial KNN graph is built on these clusters. To ensure that it is a connected graph, Kruskal’s algorithm is used to build a minimum spanning tree (MST) on the complete graph of these clusters. Finally, both the KNN graph and MST are combined to build a connected KNN graph;
- converting clones of cells into clones of nodes (clusters). Each clone will be represented by a set of nodes for the purpose of robustness and computational efficiency. For each node of each clone, the temporal labels of cells belonging to it will be preserved and the label of this node will be decided by a majority vote algorithm;
- searching for the peer nodes. Given a pair of clones, for each node of one clone, its peer node is defined as the nearest node of the closest timepoint from the other clone;
- calculating the final distance. For each node, Dijkstra's algorithm is used to compute the length of the shortest path from this node to its peer node. Within each timepoint, its distance is calculated as the sum of all nodes’ lengths divided by the total number of nodes; The final distance is calculated as the sum of each timepoint’s distance divided by the number of timepoints.

## Tutorials

* [larry_subset](https://github.com/pinellolab/MEGATRON/tree/master/docs/source/_static/notebooks/larry_subset.ipynb)
* [larry_subset_3dplot](https://github.com/pinellolab/MEGATRON/tree/master/docs/source/_static/notebooks/larry_subset_3dplot.ipynb)
* [larry](https://github.com/pinellolab/MEGATRON/tree/master/docs/source/_static/notebooks/larry.ipynb)
* [larry(using the coordinates in paper)](https://github.com/pinellolab/MEGATRON/tree/master/docs/source/_static/notebooks/larry_with_original_coordinates.ipynb)
* [celltagging_clone_traj](https://github.com/pinellolab/MEGATRON/tree/master/docs/source/_static/notebooks/celltagging_clone_traj.ipynb)
* [celltagging_clone_traj(using the coordinates in paper)](https://github.com/pinellolab/MEGATRON/tree/master/docs/source/_static/notebooks/celltagging_with_original_coordinates.ipynb)
* [mtscATAC (Lareau et al. 2021)](https://github.com/pinellolab/MEGATRON/tree/master/docs/source/_static/notebooks/mtscATAC-CD34invitro.ipynb)
