import os
import subprocess
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--in_dir", type=str, help="dataset dir")
parser.add_argument('--program', type=str, default='/home6/m_imm/cmm', help="midd eval program path")
parser.add_argument('--list', type=str, help='path to list')
parser.add_argument('--algo', type=str, help='algo name')
parser.add_argument('--z_shift', type=int, default=0, help='disp shift')


X_SIZE = 576
Y_SIZE = 384


def main():
    args = parser.parse_args()

    with open(args.list) as list_file:
        sample_dirs = [
            l.rstrip().split('_')
            for l in list_file.readlines()
        ]
    
    sum_d_err = 0
    sum_t_err = 0
    sum_mean_err = 0
    n_samples = 0
    for sample in sample_dirs:
        gt_path = os.path.join(args.in_dir, sample[0], sample[1], 'epi', 'disp_L_lidar.tif')
        pred_path = os.path.join(args.in_dir, sample[0], sample[1], 'epi', f'disp_L_{args.algo}.tif')

        params = [   
            'midd_eval', 
            pred_path, 
            gt_path, 
            '-threshold=2',
            '-mask_occl=1',
            f'-x_size={X_SIZE}',
            f'-y_size={Y_SIZE}',
            '-x_shift=0',
            '-y_shift=0',
            f'-z_shift={args.z_shift}'
        ]

        result = subprocess.run([args.program] + params, stdout=subprocess.PIPE)
        str_res = result.stdout.decode('utf-8')
        print(f'{sample[0]}_{sample[1]}: {str_res}')

        if str_res.startswith('warning'):
            continue

        vis, d_err, o_err, t_err, mean_err = map(float, str_res.split())

        print(f'{sample[0]}_{sample[1]}: {d_err} {o_err} {t_err} {mean_err}')
        sum_d_err += d_err
        sum_t_err += t_err
        sum_mean_err += mean_err
        n_samples += 1

    avg_d_err = sum_d_err / n_samples
    avg_t_err = sum_t_err / n_samples
    avg_mean_err = sum_mean_err / n_samples

    print(f'avg d_err: {avg_d_err}; avg t_err: {avg_t_err}; avg mean_err: {avg_mean_err}')


if __name__ == '__main__':
    main()