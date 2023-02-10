import os
import numpy as np
from PIL import Image
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--in_dir", type=str, help="dataset dir")
parser.add_argument("--out_dir", type=str, help="dataset dir")


def main(args):
    os.mkdir(args.out_dir)
    dirs = [d for d in os.scandir(args.in_dir) if d.is_dir()]
    for d in dirs:
        im_path = os.path.join(d.path, 'img_L.tif')
        disp_path = os.path.join(d.path, 'disp_L.tif')
        im_render_path = os.path.join(args.out_dir,  f'{d.name}_L.jpeg')
        disp_render_path = os.path.join(args.out_dir, f'{d.name}_disp_L.jpeg')

        render(im_path, im_render_path)
        render(disp_path, disp_render_path)
        print(f'Rendered in {d.name}')


def render(fin, fout):
    im = Image.open(fin)
    arr = np.array(im)
    mx = np.nanmax(arr)
    mi = np.nanmin(arr)
    print(fout)
    print(mi)
    print(mx)
    print(arr)

    arr = np.floor(arr * 255 / (mx - mi)).astype(np.uint8)
    oim = Image.fromarray(arr)
    oim.save(fout)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
