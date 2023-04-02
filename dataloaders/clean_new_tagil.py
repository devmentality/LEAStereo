import os
import sys
import shutil

import numpy as np
from PIL import Image
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--dir", type=str, help="dataset dir")
parser.add_argument("--dry_run", type=bool, default=True, help="do hiding of invalid")

required = {
    'img_L.tif',
    'img_R.tif',
    'disp_L_lidar.tif',
    'disp_R_lidar.tif',
    'disp_L_lidar0.tif',
    'disp_R_lidar0.tif'
}

REQUIRED_NO_OCC = 0.2
REQUIRED_NON_ZERO = 0.05


def is_img_valid(path: str) -> bool:
    img = Image.open(path)
    arr = np.array(img)
    all_count = arr.shape[0] * arr.shape[1]
    nonzero_frac = np.count_nonzero(arr) / all_count
    return nonzero_frac >= REQUIRED_NON_ZERO


def is_disp_valid(path: str) -> bool:
    img = Image.open(path)
    arr = np.array(img)
    all_count = arr.shape[0] * arr.shape[1]
    no_occ_frac = np.count_nonzero(~np.isnan(arr)) / all_count
    return no_occ_frac >= REQUIRED_NO_OCC


def main(args):
    dirs = [d for d in os.scandir(args.dir) if d.is_dir()]
    for i in range(len(dirs)):
        d = dirs[i]
        is_valid = (
            is_img_valid(os.path.join(d, 'img_L.tif')) and
            is_img_valid(os.path.join(d, 'img_R.tif')) and
            is_disp_valid(os.path.join(d, 'disp_L_lidar.tif')) and
            is_disp_valid(os.path.join(d, 'disp_R_lidar.tif'))
        )
        if not is_valid and not args.dry_run:
            print(f'invalid sample {d.name}. hiding dir')
            shutil.move(d, os.path.join(args.dir, f'.{d.name}'))
        elif not is_valid:
            print(f'invalid sample {d.name}. dry run')

        if i % 100 == 0:
            print(f"Handled {i} of {len(dirs)}", file=sys.stderr)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)