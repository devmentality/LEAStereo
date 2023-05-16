import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--in_dir", type=str, help="dataset dir")

def main(args):
    dirs = [d for d in os.scandir(args.in_dir) if d.is_dir()]

    for d in dirs:
        subdirs = [sd for sd in os.scandir(d) if sd.is_dir()]
        for sd in subdirs:
            res_name= os.path.join(sd, '60_midd_eval.log')
            print(f'handling {sd}')

            if os.path.exists(res_name):
                with open(res_name) as res_file:
                    lines = res_file.readlines()
                    if len(lines) >= 3:
                        header = lines[1]
                        if header.startswith('vis% d_err% o_err% t_err% mean_err'):
                            metrics = map(float, lines.split())
                            print(f'd_err: {metrics[1]}; mean_err; {metrics[3]}')
                        else:
                            print(f'file not matched. lines: {lines}')
                    else:
                        print(f'file not matched. lines: {lines}')
            else:
                print('results does not exist')

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
