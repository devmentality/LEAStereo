from __future__ import print_function
import skimage
import skimage.io
import skimage.transform
from PIL import Image

import sys
import os
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from retrain.LEAStereo import LEAStereo

from config_utils.predict_args import obtain_predict_args
from utils.colorize import get_color_map
from utils.multadds_count import count_parameters_in_MB, comp_multadds
from time import time
import matplotlib.pyplot as plt
import numpy as np
from path import Path
from dataloaders.datasets.common import read_pfm

opt = obtain_predict_args()
print(opt)

torch.backends.cudnn.benchmark = True

cuda = opt.cuda

device = 'cpu'

if cuda:
    device = 'cuda'

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

print('===> Building LEAStereo model')
model = LEAStereo(opt, device)

print('Total Params = %.2fMB' % count_parameters_in_MB(model))
print('Feature Net Params = %.2fMB' % count_parameters_in_MB(model.feature))
print('Matching Net Params = %.2fMB' % count_parameters_in_MB(model.matching))

# mult_adds = comp_multadds(model, input_size=(3, opt.crop_height, opt.crop_width))  # (3,192, 192))
# print("compute_average_flops_cost = %.2fMB" % mult_adds)

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

turbo_colormap_data = get_color_map()


def RGBToPyCmap(rgbdata):
    nsteps = rgbdata.shape[0]
    stepaxis = np.linspace(0, 1, nsteps)

    rdata = []
    gdata = []
    bdata = []
    for istep in range(nsteps):
        r = rgbdata[istep, 0]
        g = rgbdata[istep, 1]
        b = rgbdata[istep, 2]
        rdata.append((stepaxis[istep], r, r))
        gdata.append((stepaxis[istep], g, g))
        bdata.append((stepaxis[istep], b, b))

    mpl_data = {'red': rdata,
                'green': gdata,
                'blue': bdata}

    return mpl_data


# mpl_data = RGBToPyCmap(turbo_colormap_data)
# plt.register_cmap(name='turbo', data=mpl_data, lut=turbo_colormap_data.shape[0])


def save_pfm(filename, image, scale=1):
    '''
    Save a Numpy array to a PFM file.
    '''
    color = None
    file = open(filename, "w")
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    image.tofile(file)


def crop_image(image, crop_height, crop_width):
    data = np.asarray(image)
    data = np.moveaxis(data, [2], [0])
    n_layers, h, w = np.shape(data)

    if h <= crop_height and w <= crop_width:
        result = np.zeros([n_layers, crop_height, crop_width], 'float32')
        result[:, crop_height - h: crop_height, crop_width - w: crop_width] = data
    else:
        start_x = (w - crop_width) // 2
        start_y = (h - crop_height) // 2
        result = data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]

    result = np.moveaxis(result, [0], [2])
    return result


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


def test_satellite(leftname, rightname, savename, in_savename):
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
    if height <= opt.crop_height or width <= opt.crop_width:
        temp = temp[0, opt.crop_height - height: opt.crop_height, opt.crop_width - width: opt.crop_width]
    else:
        temp = temp[0, :, :]

#    skimage.io.imsave(savename, (temp * 256).astype('uint16'))
    skimage.io.imsave(savename, temp)

    left = Image.open(leftname)
    left = crop_image(left, opt.crop_height, opt.crop_width)
    skimage.io.imsave(in_savename, left)


def test(leftname, rightname, savename):
    input1, input2, height, width = test_transform(load_data(leftname, rightname), opt.crop_height, opt.crop_width)

    input1 = Variable(input1, requires_grad=False)
    input2 = Variable(input2, requires_grad=False)

    model.eval()
    if cuda:
        input1 = input1.cuda()
        input2 = input2.cuda()

    start_time = time()
    with torch.no_grad():
        prediction = model(input1, input2)

    end_time = time()

    print("Processing time: {:.4f}".format(end_time - start_time))
    temp = prediction.cpu()
    temp = temp.detach().numpy()
    if height <= opt.crop_height or width <= opt.crop_width:
        temp = temp[0, opt.crop_height - height: opt.crop_height, opt.crop_width - width: opt.crop_width]
    else:
        temp = temp[0, :, :]
    plot_disparity(savename, temp, 192)
    savename_pfm = savename.replace('png', 'pfm')
    temp = np.flipud(temp)


def plot_disparity(savename, data, max_disp):
    plt.imsave(savename, data, vmin=0, vmax=max_disp, cmap='turbo')


if __name__ == "__main__":
    print('Running')
    file_path = opt.data_path
    file_list = opt.test_list
    f = open(file_list, 'r')
    filelist = f.readlines()
    for index in range(len(filelist)):
        current_file = filelist[index]

        if opt.satellite:
            current_file = current_file[:-1]
            leftname = file_path + current_file + '/satiml.png'
            rightname = file_path + current_file + '/satimr.png'
            savename = opt.save_path + current_file + '.png'
            in_savename = opt.save_path + current_file + '_in.png'
            test_satellite(leftname, rightname, savename, in_savename)

        if opt.sceneflow:
            print(f'Running for sceneflow {index}')
            leftname = file_path + 'frames_finalpass/' + current_file[0: len(current_file) - 1]
            rightname = file_path + 'frames_finalpass/' + current_file[
                                                          0: len(current_file) - 14] + 'right/' + current_file[
                                                                                                  len(current_file) - 9:len(
                                                                                                      current_file) - 1]
            leftgtname = file_path + 'disparity/' + current_file[0: len(current_file) - 4] + 'pfm'
            print(leftgtname)
            disp_left_gt, height, width = read_pfm(leftgtname)
            savenamegt = opt.save_path + "{:d}_gt.png".format(index)
            plot_disparity(savenamegt, disp_left_gt, 192)
            savename = opt.save_path + "{:d}.png".format(index)
            test(leftname, rightname, savename)
