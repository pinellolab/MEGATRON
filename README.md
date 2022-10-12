[![CI](https://github.com/pinellolab/megatron/actions/workflows/CI.yml/badge.svg)](https://github.com/pinellolab/MEGATRON/actions/workflows/CI.yml)

# MEGATRON

**MEGA** **TR**ajectories of cl**ON**es

![megatron](./docs/source/_static/img/logo_200x204.png?raw=true)

MEGATRON, is a Python package to process and interactively visualize clonal trajectories and based on the idea of metaclones. Briefly metaclones are groups of clones that share similar transcriptomic or epigenomic profiles across time points and developmental trajectories. Based on this grouping we can create consensus trajectories i.e. trajectories that summarize similar clones with shared fates. We have tested this method on  recent lineage tracing technologies (Larry, CellTagging, profiling of mitochrondial mutations in scATAC-seq) that simultaneously track clonal relationships and transcriptional or chromatin accessibility states. 

MEGATRON also enable the dection of important genes or transcription factor binding events associated with each metaclone or diverging between two selected metaclones. Importantly, metaclones can partially overlap in the same embedding space, therefore potential enabling to discover of early events associated with cell fate not detacable by current discrete clusetering analyses.


## Installation

To install MEGATRON please...


## Datasets

* Larry [download](https://mega.nz/folder/gVhFkYaA#FH3S3VoxxeIoTW6aR-sWcA)

* Celltagging [download](https://mega.nz/folder/EJ4FXIYC#8Kx_qiPl4DTBko3AJBjufQ)

## Distances
Megatron provides a variety of algorithms to compute distance between individual clonal trajectories. Each algorithm is designed to consider the dimensions provided and quantify distances between clonal trajectories. The specifics of each approach make some better suited for different data and explain the distribution of computational load each will require. Megatron will provide users with the option to attempt different algorithms to observe which is best suited for their unique data.

In the following sections, we will define how  calculate the distance (D) between two clonal trajectories i and j. The subset of cells within i will be denoted as c_i. The subset of cells within i sequenced at time t will be denoted as c_i,t. The set of all cells within the lineage tracing data will be denoted as C.

### Centroids
As an initial naive solution, we first propose the centroids distance. Under this distance, D(i,j) is determined by identifying the Euclidean center of c_i and measuring the distance between that center and the center identified from the embedding of c_j. 

### Wasserstein
The Wasserstein distance relies on the 1-dimensional earth mover's distance (denoted as W). This distance is defined as the minimum amount of 'work' required to transform one 1-dimensional distribution into the other, where 'work' is measured as the amount of distribution weight to be moved multiplied by the distance between distributions. We reasoned that clonal trajectories that follow similar paths over time should require less work to transform their distributions into each other.

For every dimension, this distance calculates W(c_i,t, c_j,t). Then, these values are averaged over all timepoints t and all dimensions. The resulting average is D(i,j). Thus, clonal trajectories whose cells at a given timepoint are embedded adjacent to one another require less work to transform into each other and have smaller distances between them.

### Multi-Neighbor Network (MNN)
For all cells, the MNN distance first computes either a K-nearest neighbors graph or a radius graph from the embedding data (the parameters for K or exact radius should be provided by the user, although we provide sensible defaults). This neighbors network is then queried for all c_i and c_j. The set of neighbors of c_i is denoted as cn_i. We reasoned that clonal trajectories that follow similar paths over time should have neighbors belonging to the other clone.

We first calculate the fraction of cn_i,t belonging to c_j,t and the fraction of cn_j,t belonging to c_i,t. The reciprocal of these values is averaged to provide D(i,j).

### Geodesic

## Tutorials

* [larry_subset](https://github.com/pinellolab/MEGATRON/tree/master/docs/source/_static/notebooks/larry_subset.ipynb)
* [larry_subset_3dplot](https://github.com/pinellolab/MEGATRON/tree/master/docs/source/_static/notebooks/larry_subset_3dplot.ipynb)
* [larry](https://github.com/pinellolab/MEGATRON/tree/master/docs/source/_static/notebooks/larry.ipynb)
* [larry(using the coordinates in paper)](https://github.com/pinellolab/MEGATRON/tree/master/docs/source/_static/notebooks/larry_with_original_coordinates.ipynb)
* [celltagging_clone_traj](https://github.com/pinellolab/MEGATRON/tree/master/docs/source/_static/notebooks/celltagging_clone_traj.ipynb)
* [celltagging_clone_traj(using the coordinates in paper)](https://github.com/pinellolab/MEGATRON/tree/master/docs/source/_static/notebooks/celltagging_with_original_coordinates.ipynb)
