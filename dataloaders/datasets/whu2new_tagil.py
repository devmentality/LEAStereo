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


def read_left_disparity_image(file_name: str) -> np.ndarray:
    image = Image.open(file_name)
    image = image.crop((0, 0, image.size[0] - HOR_SHIFT, image.size[1]))
    data = np.asarray(image).copy()
    data[np.isnan(data)] = 999
    data += HOR_SHIFT

    return data


def read_right_disparity_image(file_name: str) -> np.ndarray:
    image = Image.open(file_name)
    image = image.crop((HOR_SHIFT, 0, image.size[0], image.size[1]))
    data = np.asarray(image).copy()
    data[np.isnan(data)] = 999
    data += HOR_SHIFT

    return data


def load_data_whu2new_tagil(data_path, current_file):
    left_name = os.path.join(data_path, current_file, 'img_L.tif')
    right_name = os.path.join(data_path, current_file, 'img_R.tif')
    disp_left_name = os.path.join(data_path, current_file, 'disp_L_lidar.tif')
    disp_right_name = os.path.join(data_path, current_file, 'disp_R_lidar.tif')

    left = read_left_image(left_name)
    right = read_right_image(right_name)
    disp_left = read_left_disparity_image(disp_left_name)
    disp_right = read_right_disparity_image(disp_right_name)

    height, width = left.shape
    left = np.transpose(np.array([left, left, left]), (1, 2, 0))
    right = np.transpose(np.array([right, right, right]), (1, 2, 0))

    print(f'Loaded sample from {current_file}, size {height} x {width}')

    temp_data = np.zeros([8, height, width], 'float32')

    set_rgb_layers(temp_data, left, right)

    temp_data[6, :, :] = disp_left
    temp_data[7, :, :] = disp_right

    return temp_data
