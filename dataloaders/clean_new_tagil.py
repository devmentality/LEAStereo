import os
import sys
import shutil

import numpy as np
from PIL import Image
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--dir", type=str, help="dataset dir")
parser.add_argument("--dry_run", type=bool, default=True, help="do hiding of invalid")
parser.add_argument("--skip_hidden", type=int, default=1, help="skip hidden sample? 0-no, 1-yes")
parser.add_argument("--white_list", type=int, default=0, help="use white list file? 0-no, 1-yes")

required = {
    'img_L.tif',
    'img_R.tif',
    'disp_L_lidar.tif',
    'disp_R_lidar.tif',
    'disp_L_lidar0.tif',
    'disp_R_lidar0.tif'
}

REQUIRED_NO_OCC = 0.3
REQUIRED_NON_ZERO = 0.8

HIGH_TH = 500
HIGH_MAX_FRAC = 0.15


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


def valid_high_frac(path: str) -> bool:
    img = Image.open(path)
    arr = np.array(img)
    high_frac = np.count_nonzero(arr > HIGH_TH) / (arr.shape[0] * arr.shape[1])
    return high_frac < HIGH_MAX_FRAC


def main(args):
    white_list = set()
    if args.white_list:
        with open('white_list.txt') as wl_file:
            for l in wl_file.readlines():
                white_list.add(l.rstrip())

    dirs = [d for d in os.scandir(args.dir) if d.is_dir()]
    for i in range(len(dirs)):
        d = dirs[i]
        if d.name.startswith('.') and args.skip_hidden:
            print(f'skipping {d.name}')
            continue

        is_valid = (
            is_img_valid(os.path.join(d, 'img_L.tif')) and
            is_img_valid(os.path.join(d, 'img_R.tif')) and
            is_disp_valid(os.path.join(d, 'disp_L_lidar.tif')) and
            is_disp_valid(os.path.join(d, 'disp_R_lidar.tif')) and
            valid_high_frac(os.path.join(d, 'img_L.tif')) and
            valid_high_frac(os.path.join(d, 'img_R.tif')) and
            (not args.white_list or d.name in white_list) 
        )

        if not is_valid and not args.dry_run:
            save_name = d.name if d.name.startswith('.') else f'.{d.name}'
            print(f'invalid sample {d.name}. hiding dir')
            shutil.move(d, os.path.join(args.dir, save_name))
        elif not is_valid:
            print(f'invalid sample {d.name}. dry run')
        elif is_valid and not args.dry_run:
            if d.name.startswith('.'):
                save_name = d.name[1:]
                print(f'valid sample {d.name}. unhiding dir')
                shutil.move(d, os.path.join(args.dir, save_name))

        if i % 100 == 0:
            print(f"Handled {i} of {len(dirs)}", file=sys.stderr)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)