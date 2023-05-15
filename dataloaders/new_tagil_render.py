import os
import numpy as np
from PIL import Image
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--in_dir", type=str, help="dataset dir")
parser.add_argument("--list", type=str, help="list of files")
parser.add_argument("--out_dir", type=str, help="dataset dir")


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    dirs = [d for d in os.scandir(args.in_dir) if d.is_dir()]

    if args.list is not None:
        sample_set = set()
        with open(args.list) as list_file:
            for line in list_file:
                if len(line.rstrip()) > 0:
                    sample_set.add(line.rstrip())

        dirs = [d for d in dirs if d.name in sample_set]
    else:
        dirs = [d for d in dirs if not d.name.startswith('.')]

    for d in dirs:
        l_path = os.path.join(d.path, 'img_L.tif')
        r_path = os.path.join(d.path, 'img_R.tif')
        disp0_path = os.path.join(d.path, 'disp_L_lidar0.tif')

        l_render_path = os.path.join(args.out_dir,  f'{d.name}_render_L.jpeg')
        r_render_path = os.path.join(args.out_dir,  f'{d.name}_render_R.jpeg')

        disp0_render_path = os.path.join(args.out_dir, f'{d.name}_render_disp_L_lidar0.jpeg')

        make_render(Image.open(l_path), 0, 250).save(l_render_path)
        make_render(Image.open(r_path), 0, 250).save(r_render_path)        
        make_disp_render(Image.open(disp0_path), 30, 250).save(disp0_render_path)

        print(f'Rendered in {d.name}')


def make_render(in_img, new_min, new_max):
    in_arr = np.asarray(in_img)
    in_min = np.min(in_arr)
    in_rng = np.max(in_arr) - np.min(in_arr)

    new_rng = new_max - new_min

    new_arr = (in_arr.astype(float) - in_min) * new_rng / in_rng + new_min
    return Image.fromarray(new_arr.astype(np.uint8))


def make_disp_render(in_disp, new_min, new_max):
    in_arr = np.asarray(in_disp).copy()
    in_min = np.nanmin(in_arr)
    in_rng = np.nanmax(in_arr) - np.nanmin(in_arr)

    new_rng = new_max - new_min

    new_arr = (in_arr - in_min) * new_rng / in_rng + new_min
    new_arr[np.isnan(new_arr)] = 0
    
    return Image.fromarray(new_arr.astype(np.uint8))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
