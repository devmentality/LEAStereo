from PIL import Image
import numpy as np
from dataclasses import dataclass


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


def test_reading():
    file_name = input()
    data = read_disparity_image(file_name)
    print(f'Transformed shape {data.disparity_map.shape}')
    n_occlusions = np.count_nonzero(data.occlusions_map)
    print(f'Occluded {n_occlusions} of {data.occlusions_map.shape[0] * data.occlusions_map.shape[1]} pixels')


if __name__ == "__main__":
    test_reading()
