import argparse


def obtain_decode_args():
    parser = argparse.ArgumentParser(description="LEStereo Decoding..")
    parser.add_argument('--dataset', type=str, default='sceneflow',
                        choices=['sceneflow', 'satellite', 'new_tagil', 'whu'],
                        help='dataset name (default: sceneflow)') 
    parser.add_argument('--step', type=int, default=3)
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    return parser.parse_args()
