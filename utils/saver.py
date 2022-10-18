import os
import sys
import shutil
import torch


class Saver:
    """
        Saves data for experiment:
        directory: run/[dataset]-[stage]/[experiment_name]
        What to save?
            - parameters: lr, epochs, crop_height, crop_width into parameters.txt
            - best checkpoints: under checkpoints/best_{checkpoint_index}.pth
            - tensorboard logs under logs/
    """
    def __init__(self, args):
        self.experiment = args.experiment if args.experiment is not None else 'default'
        self.directory = os.path.join('run', f"{args.dataset}-{args.stage}", self.experiment)
        self.logs_dir = os.path.join(self.directory, "logs")
        self.checkpoints_dir = os.path.join(self.directory, "checkpoints")

        if not os.path.exists(self.directory):
            os.makedirs(self.directory, exist_ok=False)
            os.makedirs(self.logs_dir, exist_ok=False)
            os.makedirs(self.checkpoints_dir, exist_ok=False)
        else:
            print(f"Experiment with dataset {args.dataset}, stage {args.stage} and name {self.experiment} already exists")
            sys.exit(1)

        # TODO: Should depend on stage
        with open("parameters.txt", "w") as params_file:
            params = {
                "dataset": args.dataset,
                "lr": args.lr,
                "epochs": args.epochs,
                "crop_height": args.crop_height,
                "crop_width": args.crop_width
            }

            for key in params:
                params_file.write(f"{key}:{params[key]}\n")

    def save_checkpoint(self, epoch, state, is_best):
        filename = os.path.join(self.checkpoints_dir, f"epoch_{epoch}.pth")
        torch.save(state, filename)

        if is_best:
            shutil.copyfile(filename, os.path.join(self.checkpoints_dir, 'best.pth'))

        print(f"Checkpoint saved to {filename}")
