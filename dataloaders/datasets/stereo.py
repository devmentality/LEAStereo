import os
import torch.utils.data as data
from PIL import Image
import numpy as np
from mypath import Path
import pdb
from .common import read_pfm, set_rgb_layers, train_transform, test_transform
from .satellite import load_data_satellite
from .new_tagil import load_data_new_tagil
from .whu import load_data_whu
from .whu2new_tagil import load_data_whu2new_tagil


def load_data_sceneflow(data_path, current_file):
    """
    Data layout:
        disparity
            *scene_name*
                left
                    *.png
                right
                    *.png
        frames_finalpass
            -*- same -*-
            png -> pfm

    Input:
        path of the format: frames_finalpass/*scene_name*/left/*sample_name*.png
    """
    frames_prefix = 'frames_finalpass'
    disp_prefix = 'disparity'

    scene_name = current_file.split('/', maxsplit=1)[1].rsplit('/', maxsplit=2)[0]
    sample_name = current_file.rsplit('/', maxsplit=1)[1].split('.')[0]

    left = Image.open(os.path.join(data_path, frames_prefix, scene_name, "left", f"{sample_name}.png"))
    right = Image.open(os.path.join(data_path, frames_prefix, scene_name, "right", f"{sample_name}.png"))

    disp_left, height, width = read_pfm(os.path.join(data_path, disp_prefix, scene_name, "left", f"{sample_name}.pfm"))
    disp_right, height, width = read_pfm(os.path.join(data_path, disp_prefix, scene_name, "right", f"{sample_name}.pfm"))

    size = np.shape(left)
    height = size[0]
    width = size[1]
    temp_data = np.zeros([8, height, width], 'float32')
    left = np.asarray(left)
    right = np.asarray(right)

    set_rgb_layers(temp_data, left, right)

    temp_data[6: 7, :, :] = width * 2
    temp_data[6, :, :] = disp_left
    temp_data[7, :, :] = disp_right
    return temp_data


def load_data_sceneflow_old(data_path, current_file):
    """
    Data layout:
        disparity
            35mm_forward_fast
                left
                    *.png
                right
                    *.png
        frames_finalpass
            -*- same -*-
            png -> pfm

    Input:
        file names without extension under left (like 0001)
    """
    frames_prefix = 'frames_finalpass/35mm_forward_fast/'
    disp_prefix = 'disparity/35mm_forward_fast/'

    left = Image.open(f"{data_path}{frames_prefix}left/{current_file}.png")
    right = Image.open(f"{data_path}{frames_prefix}right/{current_file}.png")

    disp_left, height, width = read_pfm(f"{data_path}{disp_prefix}left/{current_file}.pfm")
    disp_right, height, width = read_pfm(f"{data_path}{disp_prefix}right/{current_file}.pfm")

    size = np.shape(left)
    height = size[0]
    width = size[1]
    temp_data = np.zeros([8, height, width], 'float32')
    left = np.asarray(left)
    right = np.asarray(right)

    set_rgb_layers(temp_data, left, right)

    temp_data[6: 7, :, :] = width * 2
    temp_data[6, :, :] = disp_left
    temp_data[7, :, :] = disp_right
    return temp_data


def load_data_dfc2019(data_path, current_file):
    leftname = data_path + current_file + '_LEFT_RGB.tif'
    rightname = data_path + current_file + '_RIGHT_RGB.tif'
    _, sample_name = current_file.rsplit('/', maxsplit=1)
    dispname = data_path + 'Track2-Truth/' + sample_name + '_LEFT_DSP.tif'

    left = np.asarray(Image.open(leftname))
    right = np.asarray(Image.open(rightname))
    disp_left = np.asarray(Image.open(dispname))

    size = np.shape(left)

    height = size[0]
    width = size[1]
    temp_data = np.zeros([8, height, width], 'float32')

    set_rgb_layers(temp_data, left, right)

    temp_data[6: 7, :, :] = width * 2
    temp_data[6, :, :] = disp_left[:, :]
    temp = temp_data[6, :, :]
    temp[temp < 0.1] = width * 2
    temp_data[6, :, :] = temp

    return temp_data


class DatasetFromList(data.Dataset): 
    def __init__(self, args, file_list, crop_size=[256, 256], training=True, left_right=False, shift=0):
        super(DatasetFromList, self).__init__()
        self.args = args
        self.training = training
        self.crop_height = crop_size[0]
        self.crop_width = crop_size[1]
        self.left_right = left_right
        self.shift = shift

        with open(file_list, 'r') as f:
            self.file_list = f.readlines()

    def __getitem__(self, index):
        curr_file = self.file_list[index].rstrip()
        # which disparity to use
        use_left = True
        if self.args.dataset == 'sceneflow':
            temp_data = load_data_sceneflow(Path.db_root_dir('sceneflow'), curr_file)
        elif self.args.dataset == 'sceneflow_part':
            temp_data = load_data_sceneflow(Path.db_root_dir('sceneflow_part'), curr_file)
        elif self.args.dataset == 'satellite':
            temp_data = load_data_satellite(Path.db_root_dir('satellite'), curr_file)
        elif self.args.dataset == 'dfc2019':
            temp_data = load_data_dfc2019(Path.db_root_dir('dfc2019'), curr_file)
        elif self.args.dataset == 'new_tagil':
            temp_data = load_data_new_tagil(Path.db_root_dir('new_tagil'), curr_file)
        elif self.args.dataset == 'whu':
            use_left = False
            temp_data = load_data_whu(Path.db_root_dir('whu'), curr_file)

        if self.training:
            input1, input2, target = train_transform(
                temp_data, self.crop_height, self.crop_width, 
                use_left=use_left, 
                left_right=self.left_right, 
                shift=self.shift)
            return input1, input2, target
        else:
            input1, input2, target = test_transform(
                temp_data, self.crop_height, self.crop_width, 
                use_left=use_left)
            return input1, input2, target

    def __len__(self):
        return len(self.file_list)
