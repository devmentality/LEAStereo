from PIL import Image
import numpy as np
import os.path
import sys
from .common import set_rgb_layers


HOR_SHIFT = 64


def read_left_image(file_name: str) -> np.ndarray:
    img = Image.open(file_name)
    img = img.crop((0, 0, img.size[0] - HOR_SHIFT, img.size[1]))
    return np.asarray(img)


def read_right_image(file_name: str) -> np.ndarray:
    img = Image.open(file_name)
    img = img.crop((HOR_SHIFT, 0, img.size[0], img.size[1]))
    return np.asarray(img)


def read_disparity_image(file_name: str) -> np.ndarray:
    disp = Image.open(file_name)
    disp = disp.crop((0, 0, disp.size[0] - HOR_SHIFT, disp.size[1]))

    disp_arr = np.asarray(disp)
    disp_arr = (-1) * disp_arr + HOR_SHIFT

    return disp_arr


def load_data_whu(data_path, current_file):
    left_name = os.path.join(data_path, current_file, 'left.tiff')
    right_name = os.path.join(data_path, current_file, 'right.tiff')
    disp_left_name = os.path.join(data_path, current_file, 'disp_L.tiff')

    left = read_left_image(left_name)
    right = read_right_image(right_name)
    disp_left = read_disparity_image(disp_left_name)

    height, width = left.shape
    left = np.transpose(np.array([left, left, left]), (1, 2, 0))
    right = np.transpose(np.array([right, right, right]), (1, 2, 0))

    print(f'Loaded sample from {current_file}, size {height} x {width}')

    temp_data = np.zeros([8, height, width], 'float32')

    # swap images because...
    tmp = left
    left = right
    right = tmp
    set_rgb_layers(temp_data, left, right)

    # set right disparity instead
    temp_data[6, :, :] = 2 * width
    temp_data[7, :, :] = disp_left

    return temp_data

