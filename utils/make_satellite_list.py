import os
import os.path


def make_list(dataset_folder: str, list_name: str):
    root = next(os.walk(dataset_folder))
    image_folders = root[1]

    list_path = os.path.join(os.path.dirname(__file__), os.pardir, 'dataloaders', 'lists', list_name)

    with open(list_path, 'w') as list_file:
        list_file.writelines(map(lambda s: s + '\n', image_folders))


if __name__ == "__main__":
    dataset_folder = input('Dataset folder: ')
    list_name = input('List name: ')

    make_list(dataset_folder, list_name)
