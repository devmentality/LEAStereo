import os
import shutil
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--in_dir", type=str, help="source dataset dir")
parser.add_argument("--out_dir", type=str, help="destination dataset dir")


required = {
    'img_L.tif',
    'img_R.tif',
    'disp_L_lidar.tif',
    'disp_R_lidar.tif',
    'disp_L_lidar0.tif',
    'disp_R_lidar0.tif'
}


def main(args):
    os.mkdir(args.out_dir)
    dirs = [d for d in os.scandir(args.in_dir) if d.is_dir()]
    for d in dirs:
        subdirs = [sd for sd in os.scandir(d) if sd.is_dir()]
        for sd in subdirs:
            epi = os.path.join(sd, 'epi')
            if os.path.isdir(epi):
                present = set(map(str, os.listdir(epi)))
                if required.issubset(present):
                    print(f'Sample from {d.name} {sd.name} is OK')
                    sample_dir = os.path.join(args.out_dir, f'{d.name}_{sd.name}')
                    os.mkdir(sample_dir)
                    for f in required:
                        print(f'Copy {f}')
                        shutil.copy(os.path.join(epi, f), sample_dir)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)