import os
import sys
import random
from typing import List, Tuple


SEARCH_ARCH = 0.3
SEARCH_WEIGHT = 0.3
TRAIN = 0.2
VAL = 0.1


if len(sys.argv) >= 3:
    DATASET_DIR = sys.argv[1]
    LISTS_DIR = sys.argv[2]
else:
    raise Exception("Missing argument: dataset_dir, lists_dir")


def make_ranges(coeff: List[float], size: int) -> List[Tuple]:
    ranges = []
    last_end = 0
    for c in coeff:
        ranges.append((last_end, last_end + int(size * c)))
        last_end = ranges[-1][1]

    ranges.append((last_end, size))
    return ranges


def save_list(lists_dir: str, list_name: str, names: List[str]):
    list_filename = os.path.join(lists_dir, list_name)
    with open(list_filename, "w") as list_file:
        list_file.writelines(names)


def build_lists(dataset_dir: str, lists_dir: str):
    os.mkdir(lists_dir)

    sample_names = next(os.walk(dataset_dir))[1]
    random.shuffle(sample_names)

    ranges = make_ranges([
        SEARCH_ARCH,
        SEARCH_WEIGHT,
        TRAIN,
        VAL
    ], len(sample_names))

    save_list(lists_dir, "search_arch.list", sample_names[ranges[0][0]: ranges[0][1]])
    save_list(lists_dir, "search_weights.list", sample_names[ranges[1][0]: ranges[1][1]])
    save_list(lists_dir, "train.list", sample_names[ranges[2][0]: ranges[2][1]])
    save_list(lists_dir, "val.list", sample_names[ranges[3][0]: ranges[3][1]])
    save_list(lists_dir, "test.list", sample_names[ranges[4][0]: ranges[4][1]])


if __name__ == "__main__":
    build_lists(DATASET_DIR, LISTS_DIR)
