from __future__ import print_function

import sys
import shutil
import os
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from time import time
from retrain.LEAStereo import LEAStereo

from dataloaders import make_data_loader
from utils.multadds_count import count_parameters_in_MB
from utils.early_stopping import EarlyStopping
from config_utils.train_args import obtain_train_args
from torch.utils.tensorboard import SummaryWriter

opt = obtain_train_args()
print(opt)

cuda = opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

device = 'cpu'

if cuda:
    device = 'cuda'

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
kwargs = {'num_workers': opt.threads, 'pin_memory': True, 'drop_last':True}
training_data_loader, testing_data_loader = make_data_loader(opt, **kwargs)

print('===> Building model')
model = LEAStereo(opt, device)

# compute parameters
# print('Total number of model parameters : {}'.format(sum([p.data.nelement() for p in model.parameters()])))
# print('Number of Feature Net parameters: {}'.format(sum([p.data.nelement() for p in model.feature.parameters()])))
# print('Number of Matching Net parameters: {}'.format(sum([p.data.nelement() for p in model.matching.parameters()])))

print('Total Params = %.2fMB' % count_parameters_in_MB(model))
print('Feature Net Params = %.2fMB' % count_parameters_in_MB(model.feature))
print('Matching Net Params = %.2fMB' % count_parameters_in_MB(model.matching))
   
# mult_adds = comp_multadds(model, input_size=(3,opt.crop_height, opt.crop_width)) #(3,192, 192))
# print("compute_average_flops_cost = %.2fMB" % mult_adds)

if cuda:
    model = torch.nn.DataParallel(model).cuda()

torch.backends.cudnn.benchmark = True

if opt.solver == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9,0.999))
elif opt.solver == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=0.5)

if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

os.makedirs('./logs', exist_ok=True)

experiment_name = f'{opt.dataset}'
tb_dir = f'./logs/{experiment_name}'
tb_writer = SummaryWriter(tb_dir)

val_step = 0
train_step = 0


def calculate_validity_mask(target):
    # Zeros in target are occlusions
    return (target < opt.maxdisp) & (target > 0.001)


def train(epoch):
    global train_step
    epoch_loss = 0
    epoch_error = 0
    valid_iteration = 0
    
    for iteration, batch in enumerate(training_data_loader):
        input1, input2, target = Variable(batch[0], requires_grad=True), Variable(batch[1], requires_grad=True), (batch[2])

        if cuda:
            input1 = input1.cuda()
            input2 = input2.cuda()
            target = target.cuda()

        target = torch.squeeze(target, 1)
        mask = calculate_validity_mask(target)
        mask.detach_()
        valid = target[mask].size()[0]
        train_start_time = time()
        if valid > 0:
            model.train()
    
            optimizer.zero_grad()
            disp = model(input1, input2)
            loss = F.smooth_l1_loss(disp[mask], target[mask], reduction='mean')
            loss.backward()
            optimizer.step()
            
            error = torch.mean(torch.abs(disp[mask] - target[mask])) 
            train_end_time = time()
            train_time = train_end_time - train_start_time

            epoch_loss += loss.item()
            valid_iteration += 1
            epoch_error += error.item()
            print("===> Epoch[{}]({}/{}): Loss: ({:.4f}), Error: ({:.4f}), Time: ({:.2f}s)".format(epoch, iteration, len(training_data_loader), loss.item(), error.item(), train_time))
            sys.stdout.flush()

            tb_writer.add_scalar('Train loss', loss.item(), train_step + 1)
            tb_writer.add_scalar('Train error', error.item(), train_step + 1)
            train_step += 1

    print("===> Epoch {} Complete: Avg. Loss: ({:.4f}), Avg. Error: ({:.4f})".format(epoch, epoch_loss / valid_iteration, epoch_error/valid_iteration))


def val():
    global val_step
    print('Validation')
    epoch_error = 0
    valid_iteration = 0
    three_px_error_all = 0
    model.eval()
    for iteration, batch in enumerate(testing_data_loader):
        print(f'Validation iteration # {iteration}')
        input1 = Variable(batch[0], requires_grad=False)
        input2 = Variable(batch[1], requires_grad=False)
        target = Variable(batch[2], requires_grad=False)

        if cuda:
            input1 = input1.cuda()
            input2 = input2.cuda()
            target = target.cuda()

        target = torch.squeeze(target, 1)
        mask = calculate_validity_mask(target)
        mask.detach_()
        valid = target[mask].size()[0]
        print(f'Valid count = {valid}')

        if valid > 0:
            with torch.no_grad(): 
                disp = model(input1, input2)
                error = torch.mean(torch.abs(disp[mask] - target[mask])) 

                valid_iteration += 1
                epoch_error += error.item()              

                # computing 3-px error (diff < 3px or < 5%)
                predicted_disparity = disp.cpu().detach()
                true_disparity = target.cpu().detach().numpy()
                shape = true_disparity.shape
                mask = mask.cpu().detach()
                abs_diff = np.full(shape, 10000)
                abs_diff[mask] = np.abs(true_disparity[mask] - predicted_disparity[mask])
                correct = (abs_diff < 3) | (abs_diff < true_disparity * 0.05)
                three_px_error = 1 - (float(torch.sum(correct)) / float(len(np.argwhere(mask))))
                three_px_error_all += three_px_error
    
                print("===> Test({}/{}): Error: ({:.4f} {:.4f})".format(iteration, len(testing_data_loader), error.item(), three_px_error))
                sys.stdout.flush()

                tb_writer.add_scalar('Validation error', error.item(), val_step + 1)
                tb_writer.add_scalar('Validation 3px error', three_px_error, val_step + 1)
                val_step += 1

    avg_three_px_error = three_px_error_all / valid_iteration
    print("===> Test: Avg. Error: ({:.4f} {:.4f})".format(epoch_error / valid_iteration, avg_three_px_error))
    return avg_three_px_error


def train_with_early_stop():
    early_stop = EarlyStopping(opt.save_path, patience=5, delta=0.001)
    epoch = 1

    while not early_stop.early_stop:
        train(epoch)
        loss = val()
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        early_stop(loss, state, epoch)
        scheduler.step()


def save_checkpoint(save_path, epoch, state, is_best):
    filename = save_path + "epoch_{}.pth".format(epoch)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, save_path + 'best.pth')
    print("Checkpoint saved to {}".format(filename))


def train_n_epochs():
    error = 100
    for epoch in range(1, opt.nEpochs + 1):
        train(epoch)
        is_best = False
        loss = val()
        if loss < error:
            error = loss
            is_best = True
        if opt.dataset == 'sceneflow':
            if epoch >= 0:
                save_checkpoint(opt.save_path, epoch, {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best)
        else:
            if epoch % 100 == 0 and epoch >= 3000:
                save_checkpoint(opt.save_path, epoch, {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best)
            if is_best:
                save_checkpoint(opt.save_path, epoch, {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best)

        scheduler.step()

    save_checkpoint(opt.save_path, opt.nEpochs, {
        'epoch': opt.nEpochs,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, is_best)


if __name__ == '__main__':
    train_with_early_stop()
