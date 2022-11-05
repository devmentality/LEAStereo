import os

import cv2
import albumentations as aug
import sys
from dataclasses import dataclass
from typing import Any

CROP_HEIGHT = 192
CROP_WIDTH = 384
N_ITER = 10

if len(sys.argv) >= 3:
    INPUT_DIR = sys.argv[1]
    OUTPUT_DIR = sys.argv[2]
else:
    raise Exception('Missing args: input_dir output_dir')


transforms = aug.Compose(
    [
        aug.RandomCrop(always_apply=True, height=CROP_HEIGHT, width=CROP_WIDTH),
        aug.VerticalFlip(p=0.5),
        aug.RandomBrightnessContrast(always_apply=True, brightness_limit=(-0.1, 0.2), contrast_limit=(-0.1, 0.2)),
        aug.GaussNoise()
    ],
    additional_targets={
        'right': 'image',
        'disp_right': 'mask'
    }
)


@dataclass
class Sample:
    name: str
    left: Any
    right: Any
    disp_left: Any
    disp_right: Any


def dataset_images(input_dir: str):
    sample_names = next(os.walk(input_dir))[1]

    for sample_name in sample_names:
        sample = Sample(
            name=sample_name,
            left=cv2.imread(os.path.join(input_dir, sample_name, 'satiml.png')),
            right=cv2.imread(os.path.join(input_dir, sample_name, 'satimr.png')),
            disp_left=cv2.imread(os.path.join(input_dir, sample_name, 'disparityl.png')),
            disp_right=cv2.imread(os.path.join(input_dir, sample_name, 'disparityr.png')),
        )
        yield sample


def save_sample(sample: Sample, output_dir: str):
    sample_dir = os.path.join(output_dir, sample.name)
    os.mkdir(sample_dir)

    cv2.imwrite(os.path.join(sample_dir, 'satiml.png'), sample.left)
    cv2.imwrite(os.path.join(sample_dir, 'satimr.png'), sample.right)
    cv2.imwrite(os.path.join(sample_dir, 'disparityl.png'), sample.disp_left)
    cv2.imwrite(os.path.join(sample_dir, 'disparityr.png'), sample.disp_right)


def transform_dataset(input_dir: str, output_dir: str, n_iter: int):
    dataset = list(dataset_images(input_dir))

    for index in range(1, n_iter + 1):
        print(f'Processing iteration {index}')
        for sample in dataset:
            transformed = transforms(
                image=sample.left,
                right=sample.right,
                mask=sample.disp_left,
                disp_right=sample.disp_right
            )
            transformed_sample = Sample(
                name=f"{sample.name}_{index}",
                left=transformed['image'],
                right=transformed['right'],
                disp_left=transformed['mask'],
                disp_right=transformed['disp_right']
            )
            save_sample(transformed_sample, output_dir)
            print(f'Sample {sample.name} handled')


if __name__ == "__main__":
    transform_dataset(INPUT_DIR, OUTPUT_DIR, N_ITER)
