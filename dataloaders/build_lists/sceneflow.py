import os
import sys
import random
from os import path, walk


if len(sys.argv) >= 3:
    DATASET_DIR = sys.argv[1]
    LISTS_DIR = sys.argv[2]
else:
    raise Exception("Missing argument: dataset_dir, lists_dir")


train_paths = [
    path.join("frames_finalpass", "TRAIN", "A"),
    path.join("frames_finalpass", "TRAIN", "B"),
    path.join("frames_finalpass", "TRAIN", "C")
]


test_paths = [
    path.join("frames_finalpass", "TEST", "A"),
    path.join("frames_finalpass", "TEST", "B"),
    path.join("frames_finalpass", "TEST", "C")
]


def collect_left_images(dataset_dir, folder):
    print(f"Handling folder {folder}")
    full_folder_path = path.join(dataset_dir, folder)
    scenes = next(walk(full_folder_path))[1]
    image_paths = []

    for scene in scenes:
        full_scene_left_path = path.join(full_folder_path, scene, "left")
        images = next(walk(full_scene_left_path))[2]

        scene_left_path = path.join(folder, scene, "left")
        image_paths += [path.join(scene_left_path, image_path) for image_path in images]

    return image_paths


def collect_train(dateset_dir):
    train_images = []

    for folder in train_paths:
        train_images += collect_left_images(dateset_dir, folder)

    return train_images


def collect_test(dataset_dir):
    test_images = []

    for folder in test_paths:
        test_images += collect_left_images(dataset_dir, folder)

    return test_images


def save_list(lists_dir, list_name, names):
    list_filename = path.join(lists_dir, list_name)
    with open(list_filename, "w") as list_file:
        list_file.writelines([name + "\n" for name in names])


def build_lists(dataset_dir, lists_dir):
    train_images = collect_train(dataset_dir)
    random.shuffle(train_images)

    test_images = collect_test(dataset_dir)
    random.shuffle(test_images)

    os.mkdir(lists_dir)
    save_list(lists_dir, "search_arch.list", train_images[: len(train_images) // 3])
    save_list(lists_dir, "search_weights.list", train_images[len(train_images) // 3: 2 * len(train_images) // 3])
    save_list(lists_dir, "train.list", train_images[2 * len(train_images) // 3:])

    save_list(lists_dir, "val.list", test_images[: len(test_images) // 2])
    save_list(lists_dir, "test.list", test_images[len(test_images) // 2:])


if __name__ == "__main__":
    build_lists(DATASET_DIR, LISTS_DIR)
