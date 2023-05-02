import os
import re
import shutil
import numpy as np
from argparse import ArgumentParser
from PIL import Image


parser = ArgumentParser()
parser.add_argument("--in_dir", type=str, help="source dataset dir")
parser.add_argument("--out_dir", type=str, help="destination dataset dir")

HOR_SHIFT = 64


def sample_name(pfx, num):
    return f'{pfx}_{num}'


"""
  Expects structure:
    disp
      PFX_disparity_NUM.tiff
      ...
    left
      PFX_left_NUM.tiff
      ...
    right
      PFX_right_NUM.tiff
      ...
"""
def scan_dir(in_dir):
    left_dir = os.path.join(in_dir, 'left')
    right_dir = os.path.join(in_dir, 'right')
    disp_dir = os.path.join(in_dir, 'disp')

    sample_re = re.compile('([A-Z]+)_left_(\d+).tiff')
    
    lefts = os.listdir(left_dir)
    for left_name in lefts:
        match = sample_re.search(left_name)
        pfx = match.group(1)
        num = match.group(2)
        right_name = f'{pfx}_right_{num}.tiff'
        disp_name = f'{pfx}_disparity_{num}.tiff'

        yield (
            sample_name(pfx, num),
            os.path.join(left_dir, left_name),
            os.path.join(right_dir, right_name),
            os.path.join(disp_dir, disp_name)
        )    


def copy_sample(out_dir, name, left_path, right_path, disp_path): 
    sample_dir = os.path.join(out_dir, name)
    os.mkdir(sample_dir)

    shutil.copy(left_path, os.path.join(sample_dir, 'left.tiff'))
    shutil.copy(right_path, os.path.join(sample_dir, 'right.tiff'))
    shutil.copy(disp_path, os.path.join(sample_dir, 'disp_L.tiff'))


def main(args):
    os.mkdir(args.out_dir)
   
    for sample in scan_dir(args.in_dir):
        copy_sample(args.out_dir, *sample)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
