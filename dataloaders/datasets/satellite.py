from PIL import Image
import numpy as np
import os.path
from dataclasses import dataclass
from common import set_rgb_layers


def transform_to_disparity(pixel_data: np.ndarray):
    if pixel_data[0] == pixel_data[1] == pixel_data[2]:
        return pixel_data[0]
    else:
        return 0


def transform_to_occlusion(pixel_data: np.ndarray):
    if pixel_data[0] == pixel_data[1] == pixel_data[2]:
        return 0
    else:
        return 1


@dataclass
class DisparityData:
    disparity_map: np.ndarray
    occlusions_map: np.ndarray


def read_disparity_image(file_name: str) -> DisparityData:
    image = Image.open(file_name)
    data = np.asarray(image)
    disparity_map = np.apply_along_axis(transform_to_disparity, 2, data)
    occlusions_map = np.apply_along_axis(transform_to_occlusion, 2, data)

    return DisparityData(disparity_map, occlusions_map)


def load_data_satellite(data_path, current_file):
    left_name = os.path.join(data_path, current_file, 'satiml.png')
    right_name = os.path.join(data_path, current_file, 'satimr.png')
    disp_left_name = os.path.join(data_path, current_file, 'disparityl.png')
    disp_right_name = os.path.join(data_path, current_file, 'disparityr.png')

    left = Image.open(left_name)
    right = Image.open(right_name)
    disp_left_data = read_disparity_image(disp_left_name)
    disp_right_data = read_disparity_image(disp_right_name)

    height, width = left.shape

    temp_data = np.zeros([8, height, width], 'float32')

    set_rgb_layers(temp_data, left, right)

    temp_data[6, :, :] = disp_left_data.disparity_map
    temp_data[7, :, :] = disp_right_data.disparity_map

    return temp_data


def test_reading():
    file_name = input()
    data = read_disparity_image(file_name)
    print(f'Transformed shape {data.disparity_map.shape}')
    n_occlusions = np.count_nonzero(data.occlusions_map)
    print(f'Occluded {n_occlusions} of {data.occlusions_map.shape[0] * data.occlusions_map.shape[1]} pixels')


if __name__ == "__main__":
    test_reading()
