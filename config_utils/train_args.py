import argparse
from .leastereo_args import add_leastereo_args

def obtain_train_args():
    # Training settings
    parser = argparse.ArgumentParser(description='LEStereo training...')
    parser.add_argument('--maxdisp', type=int, default=192, 
                        help="max disp")
    parser.add_argument('--crop_height', type=int, required=True, 
                        help="crop height")
    parser.add_argument('--crop_width', type=int, required=True, 
                        help="crop width")
    parser.add_argument('--resume', type=str, default='', 
                        help="resume from saved model")
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='training batch size')
    parser.add_argument('--testBatchSize', type=int, default=1,
                        help='testing batch size')
    parser.add_argument('--nEpochs', type=int, default=2048, 
                        help='number of epochs to train for')
    parser.add_argument('--solver', default='adam',choices=['adam','sgd'],
                        help='solver algorithms')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning Rate. Default=0.001')
    parser.add_argument('--cuda', type=int, default=1, 
                        help='use cuda? Default=True')
    parser.add_argument('--threads', type=int, default=1, 
                        help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=2019, 
                        help='random seed to use. Default=123')
    parser.add_argument('--shift', type=int, default=0, 
                        help='random shift of left image. Default=0')
    parser.add_argument('--save_path', type=str, default='./checkpoints/', 
                        help="location to save models")
    parser.add_argument('--milestones', default=[30,50,300], metavar='N', nargs='*', 
                        help='epochs at which learning rate is divided by 2')    
    parser.add_argument('--stage', type=str, default='train', choices=['search', 'train'])
    parser.add_argument('--dataset', type=str, default='sceneflow', 
                        choices=['sceneflow', 'kitti15', 'kitti12', 'middlebury', 'sceneflow_part', 'satellite', 'dfc2019'],
                        help='dataset name')
    parser.add_argument('--experiment', type=str, default='default', help='Experiment name')

    add_leastereo_args(parser)

    args = parser.parse_args()
    return args
