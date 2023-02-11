class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'sceneflow':
            return './dataset/sceneflow/'
        elif dataset == 'kitti15':
            return './dataset/kitti2015/training/'
        elif dataset == 'kitti12':
            return './dataset/kitti2012/training/'
        elif dataset == 'middlebury':
            return './dataset/MiddEval3/trainingH/'
        elif dataset == 'sceneflow_part':
            return './dataset/sceneflow_part/'
        elif dataset == 'satellite':
            return './dataset/l2d_ntagil_20220319/'
        elif dataset == 'dfc2019':
            return './dataset/dfc2019/'
        elif dataset == 'new_tagil':
            return './dataset/new_tagil/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
