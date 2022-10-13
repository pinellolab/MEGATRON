"""The core functionality"""

from ._distance import (
    clone_distance,
    set_clone_distance,
)

from ._geodesic import build_graph, calculate_pseudotime

from ._general import (
    cluster_clones,
    subset_clones
)
from ._umap import umap

from ._stat_tests import differential_test_vars

# from ._geodesic import (
#     _average_geodesic,
#     _build_graph,
#     _cal_geodesic_dist,
#     _pairwise_geodesic_dist
# )
