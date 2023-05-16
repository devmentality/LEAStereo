import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--in_dir", type=str, help="dataset dir")

def main(args):
    dirs = [d for d in os.scandir(args.in_dir) if d.is_dir()]
    all_cnt = 0
    ok_cnt = 0

    for d in dirs:
        subdirs = [sd for sd in os.scandir(d) if sd.is_dir()]
        for sd in subdirs:
            res_name= os.path.join(sd, '60_midd_eval.log')
            print(f'handling {sd.path}')
            all_cnt += 1

            if os.path.exists(res_name):
                with open(res_name) as res_file:
                    lines = res_file.readlines()
                    if len(lines) >= 3:
                        header = lines[1]
                        if header.startswith('vis% d_err% o_err% t_err% mean_err'):
                            metrics = list(map(float, lines[2].split()))
                            ok_cnt += 1
                            print(f'vis: {metrics[0]}; d_err: {metrics[1]}; o_err: {metrics[2]}, t_err: {metrics[3]} mean_err: {metrics[4]}')
                        else:
                            print(f'file not matched. lines: {lines}')
                    else:
                        print(f'file not matched. lines: {lines}')
            else:
                print('results does not exist')
    print(f'all: {all_cnt}, ok: {ok_cnt}')
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
