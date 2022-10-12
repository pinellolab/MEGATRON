[![CI](https://github.com/pinellolab/megatron/actions/workflows/CI.yml/badge.svg)](https://github.com/pinellolab/MEGATRON/actions/workflows/CI.yml)

# MEGATRON

**MEGA** **TR**ajectories of cl**ON**es

![megatron](./docs/source/_static/img/logo_200x204.png?raw=true)

MEGATRON, is a Python package to process and interactively visualize clonal trajectories and based on the idea of metaclones. Briefly metaclones are groups of clones that share similar transcriptomic or epigenomic profiles across time points and developmental trajectories. Based on this grouping we can create consensus trajectories i.e. trajectories that summarize similar clones with shared fates. We have tested this method on  recent lineage tracing technologies (Larry, CellTagging, profiling of mitochrondial mutations in scATAC-seq) that simultaneously track clonal relationships and transcriptional or chromatin accessibility states. 

MEGATRON also enable the dection of important genes or transcription factor binding events associated with each metaclone or diverging between two selected metaclones. Importantly, metaclones can partially overlap in the same embedding space, therefore potential enabling to discover of early events associated with cell fate not detacable by current discrete clusetering analyses.


## Installation

```bash
pip install git+https://github.com/pinellolab/MEGATRON
```


## Datasets

* Larry [download](https://mega.nz/folder/gVhFkYaA#FH3S3VoxxeIoTW6aR-sWcA)

* Celltagging [download](https://mega.nz/folder/EJ4FXIYC#8Kx_qiPl4DTBko3AJBjufQ)

## Tutorials

* [larry_subset](https://github.com/pinellolab/MEGATRON/tree/master/docs/source/_static/notebooks/larry_subset.ipynb)
* [larry_subset_3dplot](https://github.com/pinellolab/MEGATRON/tree/master/docs/source/_static/notebooks/larry_subset_3dplot.ipynb)
* [larry](https://github.com/pinellolab/MEGATRON/tree/master/docs/source/_static/notebooks/larry.ipynb)
* [larry(using the coordinates in paper)](https://github.com/pinellolab/MEGATRON/tree/master/docs/source/_static/notebooks/larry_with_original_coordinates.ipynb)
* [celltagging_clone_traj](https://github.com/pinellolab/MEGATRON/tree/master/docs/source/_static/notebooks/celltagging_clone_traj.ipynb)
* [celltagging_clone_traj(using the coordinates in paper)](https://github.com/pinellolab/MEGATRON/tree/master/docs/source/_static/notebooks/celltagging_with_original_coordinates.ipynb)
