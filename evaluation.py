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
from utils.metrics import calculate_3px_error
from retrain.LEAStereo import LEAStereo
from config_utils.evaluation_args import obtain_evaluation_args
from dataloaders.datasets.stereo import load_data_dfc2019, load_data_satellite

opt = obtain_evaluation_args()

print(opt)

cuda = opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

device = 'cpu'

if cuda:
    device = 'cuda'

print('===> Building LEAStereo model')
model = LEAStereo(opt, device)

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
        checkpoint = torch.load(opt.resume, map_location=torch.device(device))

        if device == 'cpu':
            state_dict = dict()
            for key in checkpoint['state_dict'].keys():
                unwrapped_key = key.split('.', 1)[1] if key.startswith('module') else key
                state_dict[unwrapped_key] = checkpoint['state_dict'][key]
        else:
            state_dict = checkpoint['state_dict']

        model.load_state_dict(state_dict, strict=True)

    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))


def crop_image(image, crop_height, crop_width):
    data = np.asarray(image)
    data = np.moveaxis(data, [2], [0])

    result = crop_array(data, crop_height, crop_width)

    result = np.moveaxis(result, [0], [2])
    return result


def crop_array_grayscale(data, crop_height, crop_width):
    h, w = np.shape(data)

    if h <= crop_height and w <= crop_width:
        result = np.zeros([crop_height, crop_width], 'float32')
        result[crop_height - h: crop_height, crop_width - w: crop_width] = data
    else:
        start_x = (w - crop_width) // 2
        start_y = (h - crop_height) // 2
        result = data[start_y: start_y + crop_height, start_x: start_x + crop_width]

    return result


def crop_array(data, crop_height, crop_width):
    n_layers, h, w = np.shape(data)

    if h <= crop_height and w <= crop_width:
        result = np.zeros([n_layers, crop_height, crop_width], 'float32')
        result[:, crop_height - h: crop_height, crop_width - w: crop_width] = data
    else:
        start_x = (w - crop_width) // 2
        start_y = (h - crop_height) // 2
        result = data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]

    return result


def make_error_image_array(prediction, error_mask):
    h, w = np.shape(prediction)
    error_layers = np.zeros((3, h, w))
    error_layers[0, :, :] = prediction
    error_layers[1, :, :] = prediction
    error_layers[2, :, :] = prediction

    error_layers[0, error_mask] = 255
    error_layers[1, error_mask] = 0
    error_layers[2, error_mask] = 0

    return np.moveaxis(error_layers, [0], [2])


def predict(left, right):
    _, height, width = np.shape(left)
    input1 = np.ones([1, 3, opt.crop_height, opt.crop_width], 'float32')
    input1[0, :, :, :] = crop_array(left, opt.crop_height, opt.crop_width)

    input2 = np.ones([1, 3, opt.crop_height, opt.crop_width], 'float32')
    input2[0, :, :, :] = crop_array(right, opt.crop_height, opt.crop_width)

    input1 = Variable(torch.from_numpy(input1).float(), requires_grad=False)
    input2 = Variable(torch.from_numpy(input2).float(), requires_grad=False)

    model.eval()
    if cuda:
        input1 = input1.cuda()
        input2 = input2.cuda()

    with torch.no_grad():
        prediction = model(input1, input2)

    temp = prediction.cpu()
    temp = temp.detach().numpy()
    if height <= opt.crop_height or width <= opt.crop_width:
        return temp[0, opt.crop_height - height: opt.crop_height, opt.crop_width - width: opt.crop_width]
    else:
        return temp[0, :, :]


def main():
    file_path = opt.data_path
    file_list = opt.test_list
    f = open(file_list, 'r')
    filelist = f.readlines()
    avg_error = 0
    three_px_error_all = 0

    for index in range(len(filelist)):
        current_file = filelist[index][:-1]

        if opt.dfc2019:
            print(f"Running for dfc2019 {current_file}")
            data = load_data_dfc2019(file_path, current_file)
            left = data[0:3, :, :]
            right = data[3: 6, :, :]
            disp = data[6, :, :]

            _, sample_name = current_file.rsplit('/', maxsplit=1)
            savename = opt.save_path + sample_name + '.png'

            leftname = file_path + current_file + '_LEFT_RGB.tif'
            in_savename = opt.save_path + sample_name + '_in.png'
            error_savename = opt.save_path + sample_name + '_error.png'

        elif opt.satellite:
            print(f"Running for satellite {current_file}")
            data = load_data_satellite(file_path, current_file)
            left = data[0:3, :, :]
            right = data[3: 6, :, :]
            disp = data[6, :, :]

            savename = opt.save_path + current_file + '.png'

            leftname = file_path + current_file + '/satiml.png'
            in_savename = opt.save_path + current_file + '_in.png'
            error_savename = opt.save_path + current_file + '_error.png'

        else:
            raise Exception("Unsupported dataset")

        prediction = predict(left, right)
        disp = crop_array_grayscale(disp, opt.crop_height, opt.crop_width)

        mask = np.logical_and(disp >= 0.001, disp <= opt.maxdisp)
        error = np.mean(np.abs(prediction[mask] - disp[mask]))
        avg_error += error

        predicted_disparity = prediction
        true_disparity = disp

        three_px_error = calculate_3px_error(predicted_disparity, true_disparity, opt.maxdisp)
        three_px_error_all += three_px_error

        print(f"===> Frame {index}, {current_file}: EPE Error: {error}, 3px Error: {three_px_error}")

        skimage.io.imsave(savename, prediction)
        left = Image.open(leftname)
        left = crop_image(left, opt.crop_height, opt.crop_width)
        skimage.io.imsave(in_savename, left)

        error_image = make_error_image_array(prediction, ~correct & mask)
        skimage.io.imsave(error_savename, error_image)

    avg_error = avg_error / len(filelist)
    avg_three_px_error = three_px_error_all / len(filelist)
    print("===> Total {} Frames ==> AVG EPE Error: {:.4f}, AVG 3px Error: {:.4f}".format(len(filelist), avg_error, avg_three_px_error))


if __name__ == "__main__":
    main()
