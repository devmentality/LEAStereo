import os
import random
from dataclasses import dataclass
from argparse import ArgumentParser

import numpy as np
from PIL import Image
from torchvision.transforms.functional import hflip, vflip, resize

parser = ArgumentParser()
parser.add_argument("--in_dir", type=str, help="input dir")
parser.add_argument("--list", type=str, help="path to list file")
parser.add_argument("--out_dir", type=str, help="output dir")


@dataclass
class Sample:
    name: str
    left: Image
    right: Image
    displ: Image
    dispr: Image
    disp0l: Image
    disp0r: Image


def read_sample(dir, sample_name):
    left_name = os.path.join(dir, sample_name, 'img_L.tif')
    right_name = os.path.join(dir, sample_name, 'img_R.tif')
    displ_name = os.path.join(dir, sample_name, 'disp_L_lidar.tif')
    dispr_name = os.path.join(dir, sample_name, 'disp_R_lidar.tif')
    disp0l_name = os.path.join(dir, sample_name, 'disp_L_lidar0.tif')
    disp0r_name = os.path.join(dir, sample_name, 'disp_R_lidar0.tif')
    
    return Sample(
        name=sample_name,
        left=Image.open(left_name),
        right=Image.open(right_name),
        displ=Image.open(displ_name),
        dispr=Image.open(dispr_name),
        disp0l=Image.open(disp0l_name),
        disp0r=Image.open(disp0r_name)
    )


def store_sample(dir, sample):
    sample_name = sample.name + f"_{random.randint(1, 1000)}"
    os.mkdir(os.path.join(dir, sample_name))

    left_name = os.path.join(dir, sample_name, 'img_L.tif')
    right_name = os.path.join(dir, sample_name, 'img_R.tif')
    displ_name = os.path.join(dir, sample_name, 'disp_L_lidar.tif')
    dispr_name = os.path.join(dir, sample_name, 'disp_R_lidar.tif')
    disp0l_name = os.path.join(dir, sample_name, 'disp_L_lidar0.tif')
    disp0r_name = os.path.join(dir, sample_name, 'disp_R_lidar0.tif')

    sample.left.save(left_name)
    sample.right.save(right_name)
    sample.displ.save(displ_name)
    sample.dispr.save(dispr_name)
    sample.disp0l.save(disp0l_name)
    sample.disp0r.save(disp0r_name)

        
def choose(probs):
    prob_sum = 0
    i = 0
    rnd = random.random()
    for p in probs:
        prob_sum += p
        if rnd < prob_sum:
            return i
        i += 1
    return i



def aug(func):
    def wrapper(samples):
        for sample in samples:
            yield func(sample)

    return wrapper

# horizontal flip begin
def hflip_sample(sample):
    return Sample(
            name=sample.name,
            left=hflip(sample.right),
            right=hflip(sample.left),
            displ=hflip(sample.dispr),
            dispr=hflip(sample.displ),
            disp0l=hflip(sample.disp0r),
            disp0r=hflip(sample.disp0l)
        )


def hor_flip_aug(sample, flip_prob):
    c = choose([1 - flip_prob, flip_prob])
    if c == 0:
        return sample
    else:
        return hflip_sample(sample)
# horizontal flip end

# vetrical flip begin
def vflip_sample(sample):
    return Sample(
            name=sample.name,
            left=vflip(sample.left),
            right=vflip(sample.right),
            displ=vflip(sample.displ),
            dispr=vflip(sample.dispr),
            disp0l=vflip(sample.disp0l),
            disp0r=vflip(sample.disp0r)
        )


def vert_flip_aug(sample, flip_prob):
    c = choose([1 - flip_prob, flip_prob])
    if c == 0:
        return sample
    else:
        return vflip_sample(sample)
# vertical flip end

pipeline = [
    aug(lambda s: hor_flip_aug(s, 0.5)),
    aug(lambda s: vert_flip_aug(s, 0.5))
]


def run(samples, pipeline):
    for f in pipeline:
        samples = f(samples)

    return samples


def read_samples(dir, sample_names):
    for sample_name in sample_names:
        print(f'Reading {sample_name}')
        yield read_sample(dir, sample_name)


if __name__ == "__main__":
    args = parser.parse_args()
    sample_names = []
    with open(args.list) as list_file:
        for line in list_file:
            sample_names.append(line.rstrip())
    
    result = run(read_samples(args.in_dir, sample_names), pipeline)

    os.mkdir(args.out_dir)
    for sample in result:
        print('Storing sample')
        store_sample(args.out_dir, sample)
