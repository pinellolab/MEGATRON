"""Utility functions and classes"""

import numpy as np


def reorder_list(input_list,
                 ref_list):
    """Reorder a list based on a reference list

    Parameters
    ----------

    Returns
    -------
    """
    input_list = list(input_list)
    list_index = []
    for x in input_list:
        list_index.append(ref_list.index(x))
    output_list = np.array(input_list)[np.argsort(list_index)].tolist()
    return output_list
