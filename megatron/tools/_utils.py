"""Utility functions and classes"""

import numpy as np


def sort_list(input_list,
              ref_list):
    """Sort a list based on a reference list

    Parameters
    ----------

    Returns
    -------
    """
    input_set = set(input_list)
    output_list = [x for x in ref_list if x in input_set]
    return output_list
