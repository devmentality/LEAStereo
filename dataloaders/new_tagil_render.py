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

    for d in dirs:
        im_path = os.path.join(d.path, 'img_L.tif')
        disp_path = os.path.join(d.path, 'disp_L.tif')
        im_render_path = os.path.join(args.out_dir,  f'{d.name}_render_L.jpeg')
        disp_render_path = os.path.join(args.out_dir, f'{d.name}_render_disp_L.jpeg')

        render_image(im_path, im_render_path)
        render_disp(disp_path, disp_render_path)
        print(f'Rendered in {d.name}')


def render_image(fin, fout):
    im = Image.open(fin)
    arr = np.array(im)
    mx = np.nanmax(arr)
    mi = np.nanmin(arr)

    arr = np.floor(arr * 200 / (mx - mi)).astype(np.uint8)

    oim = Image.fromarray(arr)
    oim.save(fout)


def render_disp(fin, fout):
    im = Image.open(fin)
    arr = np.array(im)
    nans = np.isnan(arr)

    arr = np.floor(arr).astype(np.uint8)
    rgb_arr = np.moveaxis(np.array([arr, arr, arr]), [0], [2])
    rgb_arr[nans] = np.array([255, 0, 0])

    oim = Image.fromarray(rgb_arr)
    oim.save(fout)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
