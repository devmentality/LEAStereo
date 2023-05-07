import os
import sys
import re
import random
from typing import List, Tuple
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--in_dir", type=str, help="source dataset dir")
parser.add_argument("--out_dir", type=str, help="destination lists dir")


sample_re = re.compile('([A-Z]+)_left_(\d+).tiff')


def sample_name(left_name: str) -> str:
    match = sample_re.search(left_name)
    pfx = match.group(1)
    num = match.group(2)
    
    return f'{pfx}_{num}'


def make_list(files_dir: str) -> List[str]:
    return [
        sample_name(name)
        for name in next(os.walk(files_dir))[1]
        if not name.startswith('.')
    ]


def save_list(lists_dir: str, list_name: str, names: List[str]):
    list_filename = os.path.join(lists_dir, list_name)
    with open(list_filename, "w") as list_file:
        list_file.writelines([name + "\n" for name in names])


def build_lists(dataset_dir: str, lists_dir: str):
    os.mkdir(lists_dir)

    save_list(lists_dir, "search_arch.list", [])
    save_list(lists_dir, "search_weights.list", [])
    save_list(lists_dir, "train.list", make_list(os.path.join(dataset_dir, 'train', 'left')))
    save_list(lists_dir, "val.list", make_list(os.path.join(dataset_dir, 'val', 'left')))
    save_list(lists_dir, "test.list", make_list(os.path.join(dataset_dir, 'test', 'left')))


if __name__ == "__main__":
    args = parser.parse_args()
    build_lists(args.in_dir, args.out_dir)
