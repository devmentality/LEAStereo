from torch.utils.data import DataLoader
from dataclasses import dataclass

from dataloaders.datasets import stereo


@dataclass
class ListsSet:
    train_weights_list: str = ''
    train_arch_list: str = ''
    train_list: str = ''
    val_list: str = ''
    test_list: str = ''

    @classmethod
    def make(cls, name):
        list_prefix = f'dataloaders/lists/{name}'

        return cls(
            f'{list_prefix}/search_weights.list',
            f'{list_prefix}/search_arch.list',
            f'{list_prefix}/train.list',
            f'{list_prefix}/val.list',
            f'{list_prefix}/test.list'
        )


def make_search_data_loaders(args, **kwargs) -> (DataLoader, DataLoader, DataLoader):
    if args.stage != 'search':
        raise Exception("Invalid stage")

    list_set = ListsSet.make(args.listset)

    val_set = stereo.DatasetFromList(args, list_set.val_list,
                                     [args.crop_height, args.crop_width], training=False)
    train_weights_set = stereo.DatasetFromList(args, list_set.train_weights_list,
                                               [args.crop_height, args.crop_width], training=True)
    train_arch_set = stereo.DatasetFromList(args, list_set.train_arch_list,
                                            [args.crop_height, args.crop_width], training=True)

    val_loader = DataLoader(val_set, batch_size=args.testBatchSize, shuffle=False, **kwargs)
    train_loader_weights = DataLoader(train_weights_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    train_loader_arch = DataLoader(train_arch_set, batch_size=args.batch_size, shuffle=True, **kwargs)

    return train_loader_weights, train_loader_arch, val_loader


def make_train_data_loaders(args, **kwargs) -> (DataLoader, DataLoader):
    list_set = ListsSet.make(args.listset)

    val_set = stereo.DatasetFromList(args, list_set.val_list,
                                     [args.crop_height, args.crop_width], training=False)
    train_set = stereo.DatasetFromList(args, list_set.train_list,
                                       [args.crop_height, args.crop_width], training=True)

    val_loader = DataLoader(val_set, batch_size=args.testBatchSize, shuffle=False, **kwargs)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)

    return train_loader, val_loader
