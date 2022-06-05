from __future__ import print_function
import argparse
import skimage
import skimage.io
import skimage.transform
import sys
import os
import re
import torch
import torch.nn.parallel
import numpy as np
from PIL import Image
from struct import unpack
from torch.autograd import Variable
from utils.multadds_count import count_parameters_in_MB, comp_multadds
from retrain.LEAStereo import LEAStereo
from config_utils.evaluation_args import obtain_evaluation_args


opt = obtain_evaluation_args()

print(opt)

cuda = opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

print('===> Building LEAStereo model')
model = LEAStereo(opt)

print('Total Params = %.2fMB' % count_parameters_in_MB(model))
print('Feature Net Params = %.2fMB' % count_parameters_in_MB(model.feature))
print('Matching Net Params = %.2fMB' % count_parameters_in_MB(model.matching))

mult_adds = comp_multadds(model, input_size=(3, opt.crop_height, opt.crop_width))  # (3,192, 192))
print("compute_average_flops_cost = %.2fMB" % mult_adds)


if cuda:
    model = torch.nn.DataParallel(model).cuda()

if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))


def readPFM(file):
    with open(file, "rb") as f:
        # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
        type = f.readline().decode('latin-1')
        if "PF" in type:
            channels = 3
        elif "Pf" in type:
            channels = 1
        else:
            sys.exit(1)
        # Line 2: width height
        line = f.readline().decode('latin-1')
        width, height = re.findall('\d+', line)
        width = int(width)
        height = int(height)

        # Line 3: +ve number means big endian, negative means little endian
        line = f.readline().decode('latin-1')
        BigEndian = True
        if "-" in line:
            BigEndian = False
        # Slurp all binary data
        samples = width * height * channels;
        buffer = f.read(samples * 4)
        # Unpack floats with appropriate endianness
        if BigEndian:
            fmt = ">"
        else:
            fmt = "<"
        fmt = fmt + str(samples) + "f"
        img = unpack(fmt, buffer)
        img = np.reshape(img, (height, width))
        img = np.flipud(img)
    return img, height, width


def test_transform(temp_data, crop_height, crop_width):
    _, h, w = np.shape(temp_data)

    if h <= crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([6, crop_height, crop_width], 'float32')
        temp_data[:, crop_height - h: crop_height, crop_width - w: crop_width] = temp
    else:
        start_x = int((w - crop_width) / 2)
        start_y = int((h - crop_height) / 2)
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]
    left = np.ones([1, 3, crop_height, crop_width], 'float32')
    left[0, :, :, :] = temp_data[0: 3, :, :]
    right = np.ones([1, 3, crop_height, crop_width], 'float32')
    right[0, :, :, :] = temp_data[3: 6, :, :]
    return torch.from_numpy(left).float(), torch.from_numpy(right).float(), h, w


def load_data(leftname, rightname):
    left = Image.open(leftname)
    right = Image.open(rightname)
    size = np.shape(left)
    height = size[0]
    width = size[1]
    temp_data = np.zeros([6, height, width], 'float32')
    left = np.asarray(left)
    right = np.asarray(right)
    r = left[:, :, 0]
    g = left[:, :, 1]
    b = left[:, :, 2]
    temp_data[0, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[1, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[2, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    r = right[:, :, 0]
    g = right[:, :, 1]
    b = right[:, :, 2]
    # r,g,b,_ = right.split()
    temp_data[3, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[4, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[5, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    return temp_data


def test(leftname, rightname, savename):
    input1, input2, height, width = test_transform(load_data(leftname, rightname), opt.crop_height, opt.crop_width)
    input1 = Variable(input1, requires_grad=False)
    input2 = Variable(input2, requires_grad=False)

    model.eval()
    if cuda:
        input1 = input1.cuda()
        input2 = input2.cuda()
    with torch.no_grad():
        prediction = model(input1, input2)

    temp = prediction.cpu()
    temp = temp.detach().numpy()
    if height <= opt.crop_height and width <= opt.crop_width:
        temp = temp[0, opt.crop_height - height: opt.crop_height, opt.crop_width - width: opt.crop_width]
    else:
        temp = temp[0, :, :]
    skimage.io.imsave(savename, (temp * 256).astype('uint16'))
    return temp


if __name__ == "__main__":
    file_path = opt.data_path
    file_list = opt.test_list
    f = open(file_list, 'r')
    filelist = f.readlines()
    avg_error = 0
    avg_rate = 0
    for index in range(len(filelist)):
        current_file = filelist[index]
        if opt.kitti2015:
            leftname = file_path + 'image_2/' + current_file[0: len(current_file) - 1]
            rightname = file_path + 'image_3/' + current_file[0: len(current_file) - 1]
            dispname = file_path + 'disp_occ_0/' + current_file[0: len(current_file) - 1]
            savename = opt.save_path + current_file[0: len(current_file) - 1]
            disp = Image.open(dispname)
            disp = np.asarray(disp) / 256.0
        elif opt.kitti:
            leftname = file_path + 'colored_0/' + current_file[0: len(current_file) - 1]
            rightname = file_path + 'colored_1/' + current_file[0: len(current_file) - 1]
            dispname = file_path + 'disp_occ/' + current_file[0: len(current_file) - 1]
            savename = opt.save_path + current_file[0: len(current_file) - 1]
            disp = Image.open(dispname)
            disp = np.asarray(disp) / 256.0

        else:
            leftname = opt.data_path + 'frames_finalpass/' + current_file[0: len(current_file) - 1]
            rightname = opt.data_path + 'frames_finalpass/' + current_file[
                                                              0: len(current_file) - 14] + 'right/' + current_file[
                                                                                                      len(current_file) - 9:len(
                                                                                                          current_file) - 1]
            dispname = opt.data_path + 'disparity/' + current_file[0: len(current_file) - 4] + 'pfm'
            savename = opt.save_path + str(index) + '.png'
            disp, height, width = readPFM(dispname)

        prediction = test(leftname, rightname, savename)
        mask = np.logical_and(disp >= 0.001, disp <= opt.max_disp)

        error = np.mean(np.abs(prediction[mask] - disp[mask]))
        rate = np.sum(np.abs(prediction[mask] - disp[mask]) > opt.threshold) / np.sum(mask)
        avg_error += error
        avg_rate += rate
        print("===> Frame {}: ".format(index) + current_file[0:len(
            current_file) - 1] + " ==> EPE Error: {:.4f}, Error Rate: {:.4f}".format(error, rate))
    avg_error = avg_error / len(filelist)
    avg_rate = avg_rate / len(filelist)
    print("===> Total {} Frames ==> AVG EPE Error: {:.4f}, AVG Error Rate: {:.4f}".format(len(filelist), avg_error,
                                                                                          avg_rate))
