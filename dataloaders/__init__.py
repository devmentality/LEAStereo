from torch.utils.data import DataLoader
from dataloaders.datasets import stereo


def make_data_loader(args, **kwargs):
    if args.dataset == 'satellite':
        if args.stage != 'train':
            raise Exception('Stages other than train are not supported')

        train_list = 'dataloaders/lists/satellite_train.list'
        test_list = 'dataloaders/lists/satellite_test.list'

        train_set = stereo.DatasetFromList(args, train_list, [args.crop_height, args.crop_width], True)
        test_set = stereo.DatasetFromList(args, test_list, [args.crop_height, args.crop_width], False)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.testBatchSize, shuffle=False, **kwargs)

        return train_loader, test_loader

    if args.dataset == 'dfc2019':
        if args.stage != 'train':
            raise Exception('Stages other than train are not supported')

        train_list = 'dataloaders/lists/dfc2_train.list'
        test_list = 'dataloaders/lists/dfc2_test.list'

        train_set = stereo.DatasetFromList(args, train_list, [args.crop_height, args.crop_width], True)
        test_set = stereo.DatasetFromList(args, test_list, [args.crop_height, args.crop_width], False)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.testBatchSize, shuffle=False, **kwargs)

        return train_loader, test_loader

    ####### custom sceneflow part ##########
    if args.dataset == 'sceneflow_part':
        if args.stage != 'train':
            raise Exception('Stages other than train are not supported')

        train_list = 'dataloaders/lists/sf_part_train.list'
        test_list = 'dataloaders/lists/sf_part_test.list'

        train_set = stereo.DatasetFromList(args, train_list, [args.crop_height, args.crop_width], True)
        test_set = stereo.DatasetFromList(args, test_list,  [576, 960], False)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.testBatchSize, shuffle=False, **kwargs)

        return train_loader, test_loader

    # SceneFlow
    elif args.dataset == 'sceneflow':
        train_weights_list = 'dataloaders/lists/sceneflow_search_train_weights.list'  # randomly select 10,000 from the original training set
        train_arch_list = 'dataloaders/lists/sceneflow_search_train_arch.list'  # randomly select 10,000 from the original training set
        val_list = 'dataloaders/lists/sceneflow_search_val.list'  # randomly select 1,000 from the original test set
        train_list = 'dataloaders/lists/sceneflow_train.list'  # original training set: 35,454
        test_list = 'dataloaders/lists/sceneflow_test.list'  # original test set:4,370

        train_weights_set = stereo.DatasetFromList(args, train_weights_list, [args.crop_height, args.crop_width], True)
        train_arch_set = stereo.DatasetFromList(args, train_arch_list, [args.crop_height, args.crop_width], True)
        train_set = stereo.DatasetFromList(args, train_list, [args.crop_height, args.crop_width], True)
        val_set = stereo.DatasetFromList(args, val_list,  [576, 960], False)
        test_set = stereo.DatasetFromList(args, test_list,  [576, 960], False)

        val_loader = DataLoader(val_set, batch_size=args.testBatchSize, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.testBatchSize, shuffle=False, **kwargs)

        if args.stage == 'search':
            train_loader_weights = DataLoader(train_weights_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            train_loader_arch = DataLoader(train_arch_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            return train_loader_weights, train_loader_arch, val_loader, test_loader

        elif args.stage == 'train':
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            return train_loader, test_loader
        else:
            raise Exception('Unrecognized stage')
    else:
        raise NotImplementedError
