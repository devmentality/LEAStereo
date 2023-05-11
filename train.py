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

from dataloaders.make_data_loaders import make_train_data_loaders
from utils.multadds_count import count_parameters_in_MB
from utils.early_stopping import EarlyStopping
from config_utils.train_args import obtain_train_args
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import calculate_3px_error
from edge_detection.edge_detection import gradient_aware_loss

opt = obtain_train_args()
print(opt)
print(f"Running experiment {opt.experiment}")

# Prepare dir
run_dir = f"{opt.dataset}/{opt.experiment}/"
os.makedirs(opt.save_path + run_dir, exist_ok=True)
opt.save_path = opt.save_path + run_dir

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
training_data_loader, val_data_loader = make_train_data_loaders(opt, **kwargs)

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

tb_dir = f'./logs/{opt.experiment}'
tb_writer = SummaryWriter(tb_dir)

val_step = 0
train_step = 0


def combined_loss(predict, target, mask, grad_loss_w):
    direct_loss = F.smooth_l1_loss(predict[mask], target[mask], reduction='mean')
    grad_loss = gradient_aware_loss(predict, target, device=device)
    comb_loss = direct_loss + grad_loss * grad_loss_w 

    print(f'Direct loss: {direct_loss.item()}, Gradient loss: {grad_loss.item()}, Combined loss: {comb_loss.item()}')
    return combined_loss


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
            #loss = F.smooth_l1_loss(disp[mask], target[mask], reduction='mean')
            loss = combined_loss(disp, target, mask, opt.edge_loss_w)
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
            tb_writer.add_scalar('Train EPE', error.item(), train_step + 1)
            train_step += 1

    tb_writer.add_scalar('Train Epoch EPE', epoch_error/valid_iteration, epoch)

    print("===> Epoch {} Complete: Avg. Loss: ({:.4f}), Avg. Error: ({:.4f})".format(epoch, epoch_loss / valid_iteration, epoch_error/valid_iteration))


def val(epoch):
    global val_step
    print('Validation')
    epoch_error = 0
    valid_iteration = 0
    three_px_error_all = 0
    model.eval()
    for iteration, batch in enumerate(val_data_loader):
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
                predicted_disparity = disp.cpu().detach().numpy()
                true_disparity = target.cpu().detach().numpy()
                three_px_error = calculate_3px_error(predicted_disparity, true_disparity, opt.maxdisp)
                three_px_error_all += three_px_error
    
                print("===> Test({}/{}): Error: ({:.4f} {:.4f})".format(iteration, len(val_data_loader), error.item(), three_px_error))
                sys.stdout.flush()

                tb_writer.add_scalar('Validation EPE', error.item(), val_step + 1)
                tb_writer.add_scalar('Validation 3px error', three_px_error, val_step + 1)
                val_step += 1

    avg_three_px_error = three_px_error_all / valid_iteration
    avg_epe = epoch_error / valid_iteration
    tb_writer.add_scalar('Validation Epoch EPE', avg_epe, epoch)
    tb_writer.add_scalar('Validation Epoch 3px error', avg_three_px_error, epoch)

    print("===> Test: Avg. Error: ({:.4f} {:.4f})".format(avg_epe, avg_three_px_error))
    return avg_three_px_error


def train_with_early_stop():
    early_stop = EarlyStopping(opt.save_path, patience=30, delta=0.001)
    epoch = 1

    while not early_stop.early_stop:
        train(epoch)
        loss = val(epoch)
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        early_stop(loss, state, epoch)
        scheduler.step()
        epoch += 1


def save_checkpoint(save_path, epoch, state, is_best):
    filename = save_path + f"epoch_{epoch}.pth"
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, save_path + 'best.pth')
    print("Checkpoint saved to {}".format(filename))


def train_n_epochs():
    error = 100
    for epoch in range(1, opt.nEpochs + 1):
        train(epoch)
        is_best = False
        loss = val(epoch)
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
