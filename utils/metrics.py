import numpy as np
from numpy import ndarray
from typing import Tuple, Any


def calculate_validity_mask(target: ndarray, max_disp: int):
    # Zeros in target are occlusions
    return (target < max_disp) & (target > 0.001)


def calculate_3px_error(predicted_disparity: ndarray, true_disparity: ndarray, max_disp: int) -> float:
    """ Computing 3-px error (diff < 3px or < 5%) """
    inf_disp = 10000
    shape = true_disparity.shape
    mask = calculate_validity_mask(true_disparity, max_disp)
    abs_diff = np.full(shape, inf_disp)
    abs_diff[mask] = np.abs(true_disparity[mask] - predicted_disparity[mask])
    correct = (abs_diff < 3) | (abs_diff < true_disparity * 0.05)
    three_px_error = 1 - (float(np.sum(correct)) / float(len(np.argwhere(mask))))

    return three_px_error


def calculate_3px_error_and_correct_mask(predicted_disparity: ndarray, true_disparity: ndarray, max_disp: int) -> Tuple[float, Any]:
    """ Computing 3-px error (diff < 3px or < 5%) """
    inf_disp = 10000
    shape = true_disparity.shape
    mask = calculate_validity_mask(true_disparity, max_disp)
    abs_diff = np.full(shape, inf_disp)
    abs_diff[mask] = np.abs(true_disparity[mask] - predicted_disparity[mask])
    correct = (abs_diff < 3) | (abs_diff < true_disparity * 0.05)
    three_px_error = 1 - (float(np.sum(correct)) / float(len(np.argwhere(mask))))

    return three_px_error, correct


def calculate_bad_pixel_frac(predicted_disparity: ndarray, true_disparity: ndarray, max_disp: int, threshold: int) -> float:
    inf_disp = 10000
    shape = true_disparity.shape
    mask = calculate_validity_mask(true_disparity, max_disp)
    abs_diff = np.full(shape, inf_disp)
    abs_diff[mask] = np.abs(true_disparity[mask] - predicted_disparity[mask])
    correct = (abs_diff <= threshold)
    bad_pixel_frac = 1 - (float(np.sum(correct)) / float(len(np.argwhere(mask))))

    return bad_pixel_frac
