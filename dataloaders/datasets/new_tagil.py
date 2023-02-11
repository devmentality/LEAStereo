from PIL import Image
import numpy as np
import os.path
from .common import set_rgb_layers


def read_disparity_image(file_name: str) -> np.ndarray:
    image = Image.open(file_name)
    data = np.asarray(image)
    disparity_map = np.nan_to_num(data)

    return disparity_map


def load_data_new_tagil(data_path, current_file):
    left_name = os.path.join(data_path, current_file, 'img_L.tif')
    right_name = os.path.join(data_path, current_file, 'img_R.tif')
    disp_left_name = os.path.join(data_path, current_file, 'disp_L.tif')
    disp_right_name = os.path.join(data_path, current_file, 'disp_R.tif')

    left = np.asarray(Image.open(left_name))
    right = np.asarray(Image.open(right_name))
    disp_left = read_disparity_image(disp_left_name)
    disp_right = read_disparity_image(disp_right_name)

    height, width = left.shape
    left = np.array([left, left, left])
    right = np.array([right, right, right])

    print(f'Loaded sample from {current_file}, size {height} x {width}')

    temp_data = np.zeros([8, height, width], 'float32')

    set_rgb_layers(temp_data, left, right)

    temp_data[6, :, :] = disp_left
    temp_data[7, :, :] = disp_right

    return temp_data


def test_reading():
    file_name = input()
    data = read_disparity_image(file_name)
    print(f'Transformed shape {data.shape}')
    n_all = data.shape[0] * data.shape[1]
    n_noocclusions = np.count_nonzero(data)
    print(f'Occluded {n_all - n_noocclusions} of {n_all} pixels')


if __name__ == "__main__":
    test_reading()
