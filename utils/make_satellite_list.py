import os
import os.path


def make_list(dataset_folder: str, list_name_prefix: str):
    root = next(os.walk(dataset_folder))
    image_folders = root[1]

    train_part = 0.9
    split_idx = int(len(image_folders) * train_part)

    train_set = image_folders[:split_idx]
    test_set = image_folders[split_idx:]

    print(f'Train set size: {len(train_set)}')
    print(f'Test set size: {len(test_set)}')

    train_list_path = os.path.join(
        os.path.dirname(__file__), os.pardir,
        'dataloaders', 'lists', f'{list_name_prefix}_train.list')

    test_list_path = os.path.join(
        os.path.dirname(__file__), os.pardir,
        'dataloaders', 'lists', f'{list_name_prefix}_test.list')

    with open(train_list_path, 'w') as list_file:
        list_file.writelines(map(lambda s: s + '\n', train_set))

    with open(test_list_path, 'w') as list_file:
        list_file.writelines(map(lambda s: s + '\n', test_set))


if __name__ == "__main__":
    dataset_folder = input('Dataset folder: ')
    list_name = input('List name prefix: ')

    make_list(dataset_folder, list_name)
