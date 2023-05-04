import argparse
from .leastereo_args import add_leastereo_args


def obtain_evaluation_args():
    parser = argparse.ArgumentParser(description='LEStereo Prediction')
    parser.add_argument('--crop_height', type=int, required=True, help="crop height")
    parser.add_argument('--crop_width', type=int, required=True, help="crop width")
    parser.add_argument('--maxdisp', type=int, default=192, help="max disp")
    parser.add_argument('--resume', type=str, default='', help="resume from saved model")
    parser.add_argument('--cuda', type=bool, default=False, help='use cuda?')
    parser.add_argument('--data_path', type=str, required=True, help="data root")
    parser.add_argument('--test_list', type=str, required=True, help="training list")
    parser.add_argument('--save_path', type=str, default='./result/', help="location to save result")
    parser.add_argument('--threshold', type=float, default=3.0, help="threshold of error rates")

    parser.add_argument('--sceneflow', type=int, default=0, help='sceneflow dataset? Default=False')
    parser.add_argument('--kitti2012', type=int, default=0, help='kitti 2012? Default=False')
    parser.add_argument('--kitti2015', type=int, default=0, help='kitti 2015? Default=False')
    parser.add_argument('--middlebury', type=int, default=0, help='Middlebury? Default=False')
    parser.add_argument('--satellite', type=int, default=0, help='Satellite? Default=False')
    parser.add_argument('--dfc2019', type=int, default=0, help='DFC2019?')
    parser.add_argument('--new_tagil', type=int, default=0, help='New Tagil?')
    parser.add_argument('--whu', type=int, default=0, help='WHU?')

    add_leastereo_args(parser)

    args = parser.parse_args()
    return args
