import numpy as np
from PIL import Image
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--fin", type=str, help="src image")
parser.add_argument("--fout", type=str, help="dst image")


def main(args):
    im = Image.open(args.fin)
    arr = np.array(im)
    mx = np.max(arr)
    mi = np.min(arr)
    arr = np.floor(arr * 255 / (mx - mi)).astype(np.uint8)

    print(arr)
    oim = Image.fromarray(arr)
    oim.save(args.fout)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
