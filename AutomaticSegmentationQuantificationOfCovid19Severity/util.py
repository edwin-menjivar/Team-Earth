"""
Created by Joseph Edradan
Github: https://github.com/josephedradan

Date created: 5/10/2022

Purpose:

Details:

Description:

Notes:

IMPORTANT NOTES:

Explanation:

Tags:

Reference:

"""
from typing import Tuple

import numpy as np


def get_np_ndarray_binary_using_threshold(np_ndarray_given: np.ndarray, threshold=0) -> np.ndarray:
    """
    Notes:
        With higher dimensional arrays, the conditional is applied over all values regardless of dimension

    :param np_ndarray_given:
    :param threshold:
    :return:
    """
    np_ndarray_copy = np_ndarray_given.copy()

    np_ndarray_copy[np_ndarray_copy > threshold] = 1

    return np_ndarray_copy


def set_np_ndarray_binary_using_threshold(np_ndarray_given: np.ndarray, threshold=0) -> np.ndarray:
    """
    Notes:
        With higher dimensional arrays, the conditional is applied over all values regardless of dimension

    :param np_ndarray_given:
    :param threshold:
    :return:
    """
    np_ndarray_given[np_ndarray_given > threshold] = 1
    return np_ndarray_given


def calculate_count_region_sub_over_count_region_total(np_ndarray_region_total: np.ndarray,
                                                       np_ndarray_region_sub: np.ndarray,
                                                       threshold=0) -> Tuple[float, int, int]:
    """
    Notes:
        Calculate value of
            (count of region) / (count of total)

    :param np_ndarray_region_total:
    :param np_ndarray_region_sub:
    :param threshold:
    :return:
    """

    counter_region_total = np.unique(get_np_ndarray_binary_using_threshold(np_ndarray_region_total,
                                                                           threshold),
                                     return_counts=True)

    value_region_total_non, value_region_total = counter_region_total[0]
    count_region_total_non, count_region_total = counter_region_total[1]

    counter_region_sub = np.unique(get_np_ndarray_binary_using_threshold(np_ndarray_region_sub,
                                                                         threshold
                                                                         ),
                                   return_counts=True)

    value_region_sub_non, value_region_sub = counter_region_sub[0]
    count_region_sub_non, count_region_sub = counter_region_sub[1]

    fraction = count_region_sub / count_region_total

    return fraction, count_region_total, count_region_sub
