import torch.utils.data as data
from PIL import Image
import numpy as np
from mypath import Path
import pdb
from .common import read_pfm, set_rgb_layers, train_transform, test_transform
from .satellite import load_data_satellite


def load_data_sceneflow(data_path, current_file):
    A = current_file
    filename = data_path + 'frames_finalpass/' + A[0: len(A) - 1]
    left = Image.open(filename)
    filename = data_path + 'frames_finalpass/' + A[0: len(A) - 14] + 'right/' + A[len(A) - 9:len(A) - 1]
    right = Image.open(filename)
    filename = data_path + 'disparity/' + A[0: len(A) - 4] + 'pfm'
    disp_left, height, width = read_pfm(filename)
    filename = data_path + 'disparity/' + A[0: len(A) - 14] + 'right/' + A[len(A) - 9: len(A) - 4] + 'pfm'
    disp_right, height, width = read_pfm(filename)
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


def load_kitti2015_data(file_path, current_file):
    """ load current file from the list"""
    filename = file_path + 'image_2/' + current_file[0: len(current_file) - 1]
    left = Image.open(filename)
    filename = file_path + 'image_3/' + current_file[0: len(current_file) - 1]
    right = Image.open(filename)
    filename = file_path + 'disp_occ_0/' + current_file[0: len(current_file) - 1]
    disp_left = Image.open(filename)
    temp = np.asarray(disp_left)
    size = np.shape(left)

    height = size[0]
    width = size[1]
    temp_data = np.zeros([8, height, width], 'float32')
    left = np.asarray(left)
    right = np.asarray(right)
    disp_left = np.asarray(disp_left)

    set_rgb_layers(temp_data, left, right)

    temp_data[6: 7, :, :] = width * 2
    temp_data[6, :, :] = disp_left[:, :]
    temp = temp_data[6, :, :]
    temp[temp < 0.1] = width * 2 * 256
    temp_data[6, :, :] = temp / 256.
    
    return temp_data


def load_kitti2012_data(file_path, current_file):
    """ load current file from the list"""
    filename = file_path + 'colored_0/' + current_file[0: len(current_file) - 1]
    left = Image.open(filename)
    filename = file_path+'colored_1/' + current_file[0: len(current_file) - 1]
    right = Image.open(filename)
    filename = file_path+'disp_noc/' + current_file[0: len(current_file) - 1]  # disp_occ

    disp_left = Image.open(filename)
    temp = np.asarray(disp_left)
    size = np.shape(left)

    height = size[0]
    width = size[1]
    temp_data = np.zeros([8, height, width], 'float32')
    left = np.asarray(left)
    right = np.asarray(right)
    disp_left = np.asarray(disp_left)

    set_rgb_layers(temp_data, left, right)

    temp_data[6: 7, :, :] = width * 2
    temp_data[6, :, :] = disp_left[:, :]
    temp = temp_data[6, :, :]
    temp[temp < 0.1] = width * 2 * 256
    temp_data[6, :, :] = temp / 256.
    
    return temp_data


def load_data_md(file_path, current_file, eth=False):
    """ load current file from the list"""
    imgl = file_path + current_file[0: len(current_file) - 1]
    gt_l = imgl.replace('im0.png', 'disp0GT.pfm')
    imgr = imgl.replace('im0.png', 'im1.png')

    left = Image.open(imgl)
    right = Image.open(imgr)

    disp_left, height, width = read_pfm(gt_l)
    pdb.set_trace()

    temp_data = np.zeros([8, height, width], 'float32')
    left = np.asarray(left)
    right = np.asarray(right)
    disp_left = np.asarray(disp_left)

    set_rgb_layers(temp_data, left, right)

    temp_data[6: 7, :, :] = width * 2
    temp_data[6, :, :] = disp_left[:, :]
    temp = temp_data[6, :, :]
    temp[temp < 0.1] = width * 2 * 256
    temp_data[6, :, :] = temp #/ 256.
    
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
        if self.args.dataset == 'kitti12':
            temp_data = load_kitti2012_data(Path.db_root_dir('kitti12'), self.file_list[index])
        elif self.args.dataset == 'kitti15':
            temp_data = load_kitti2015_data(Path.db_root_dir('kitti15'), self.file_list[index])
        elif self.args.dataset == 'sceneflow':
            temp_data = load_data_sceneflow(Path.db_root_dir('sceneflow'), self.file_list[index])  
        elif self.args.dataset == 'middlebury':
            temp_data = load_data_md(Path.db_root_dir('middlebury'), self.file_list[index])
        elif self.args.dataset == 'sceneflow_part':
            temp_data = load_data_sceneflow(Path.db_root_dir('sceneflow_part'), self.file_list[index])
        elif self.args.dataset == 'satellite':
            curr_file = self.file_list[index][:-1]
            temp_data = load_data_satellite(Path.db_root_dir('satellite'), curr_file)
        elif self.args.dataset == 'dfc2019':
            curr_file = self.file_list[index][:-1]
            temp_data = load_data_dfc2019(Path.db_root_dir('dfc2019'), curr_file)

        if self.training:
            input1, input2, target = train_transform(temp_data, self.crop_height, self.crop_width, self.left_right, self.shift)
            return input1, input2, target
        else:
            input1, input2, target = test_transform(temp_data, self.crop_height, self.crop_width)
            return input1, input2, target

    def __len__(self):
        return len(self.file_list)
