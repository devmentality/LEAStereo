import os
import subprocess
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--in_dir", type=str, help="eval dir")
parser.add_argument('--list', type=str, help='path to list')


def main():
    args = parser.parse_args()

    with open(args.list) as list_file:
        sample_dirs = [
            l.rstrip()
            for l in list_file.readlines()
        ]
    
    sum_bad1 = 0
    sum_bad2 = 0
    sum_mean_err = 0
    n_samples = 0
    for sample in sample_dirs:
        metrics_path = os.path.join(args.in_dir, f'{sample}_metrics.txt')

        with open(metrics_path) as metrics_file:
            epe, d1, bad2, bad1 = map(float, metrics_file.readlines()[1].split())

        print(f'{sample[0]}_{sample[1]}: {epe} {d1} {bad2} {bad1}')
        sum_bad1 += bad1
        sum_bad2 += bad2
        sum_mean_err += epe
        n_samples += 1

    avg_bad1 = sum_bad1 / n_samples
    avg_bad2 = sum_bad2 / n_samples
    avg_mean_err = sum_mean_err / n_samples

    print(f'avg epe: {avg_mean_err}; avg bad2: {avg_bad2}; avg bad1: {avg_bad1}')


if __name__ == '__main__':
    main()