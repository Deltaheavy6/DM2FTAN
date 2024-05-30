# coding: utf-8
import argparse
import os
import datetime
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
# import torch.backends.cudnn
from torch.utils.data import _utils



from model import DM2FNet,MAmba2Net,DM2FNetTAN
from tools.config import TRAIN_ITS_ROOT, TEST_SOTS_ROOT
# from tools.config import OHAZE_ROOT, HazeRD_ROOT
from datasets import ItsDataset, SotsDataset,OHazeDataset,random_crop
from tools.utils import AvgMeter, check_mkdir

from skimage.metrics import peak_signal_noise_ratio, structural_similarity,mean_squared_error

model_Sel= DM2FNetTAN

def parse_args():
    parser = argparse.ArgumentParser(description='Train a DM2FNet')
    parser.add_argument(
        '--OHaze',type=str, default='./data/ITS',\
            help="the training directory"
    )
    parser.add_argument(
        '--gpus', type=str, default='0', help='gpus to use ')
    parser.add_argument('--ckpt-path', default='./ckpt', help='checkpoint path')
    parser.add_argument(
        '--exp-name',
        default='RESIDE_ITS',
        help='experiment name.')
    args = parser.parse_args()

    return args


cfgs = {
    'use_physical': True,
    'iter_num': 80000,
    'train_batch_size': 32,
    'last_iter': 0,
    'lr': 5e-4,
    'lr_decay': 0.90,
    
    'weight_decay': 0,
    'momentum': 0.9,
    'snapshot': '',
    'val_freq': 1000,
    'crop_size': 256
}


def main():
    net = model_Sel().cuda().train()
    # net = nn.DataParallel(net)

    optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters()
                    if name[-4:] == 'bias' and param.requires_grad],
         'lr': 2 * cfgs['lr']},
        {'params': [param for name, param in net.named_parameters()
                    if name[-4:] != 'bias' and param.requires_grad],
         'lr': cfgs['lr'], 'weight_decay': cfgs['weight_decay']}
    ])

    if len(cfgs['snapshot']) > 0:
        print('training resumes from \'%s\'' % cfgs['snapshot'])
        net.load_state_dict(torch.load(os.path.join(args.ckpt_path,
                                                    args.exp_name, cfgs['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(args.ckpt_path,
                                                          args.exp_name, cfgs['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * cfgs['lr']
        optimizer.param_groups[1]['lr'] = cfgs['lr']

    check_mkdir(args.ckpt_path)
    check_mkdir(os.path.join(args.ckpt_path, args.exp_name))
    open(log_path, 'w').write(str(cfgs) + '\n\n')

    train(net, optimizer)


def train(net, optimizer):
    curr_iter = cfgs['last_iter']

    while curr_iter <= cfgs['iter_num']:
        train_loss_record = AvgMeter()
        loss_x_jf_record, loss_x_j0_record = AvgMeter(), AvgMeter()
        loss_x_j1_record, loss_x_j2_record = AvgMeter(), AvgMeter()
        loss_x_j3_record, loss_x_j4_record = AvgMeter(), AvgMeter()
        loss_x_j5_record=  AvgMeter()
        loss_t_record, loss_a_record = AvgMeter(), AvgMeter()
        
        for data in train_loader:
            # print(data)
            optimizer.param_groups[0]['lr'] = 2 * cfgs['lr'] * (1 - float(curr_iter) / cfgs['iter_num']) \
                                              ** cfgs['lr_decay']
            optimizer.param_groups[1]['lr'] = cfgs['lr'] * (1 - float(curr_iter) / cfgs['iter_num']) \
                                              ** cfgs['lr_decay']

            haze, gt_trans_map, gt_ato, gt, _ = data

            batch_size = haze.size(0)

            haze = haze.cuda()
            gt_trans_map = gt_trans_map.cuda()
            gt_ato = gt_ato.cuda()
            gt = gt.cuda()

            optimizer.zero_grad()

            x_jf, x_j0, x_j1, x_j2, x_j3, x_j4,x_j5, t, a = net(haze)

            loss_x_jf = criterion(x_jf, gt)
            loss_x_j0 = criterion(x_j0, gt)
            loss_x_j1 = criterion(x_j1, gt)
            loss_x_j2 = criterion(x_j2, gt)
            loss_x_j3 = criterion(x_j3, gt)
            loss_x_j4 = criterion(x_j4, gt)
            loss_x_j5 = criterion(x_j5, gt)

            loss_t = criterion(t, gt_trans_map)
            loss_a = criterion(a, gt_ato)

            loss = loss_x_jf + loss_x_j0 + loss_x_j1 + loss_x_j2 + loss_x_j3 + loss_x_j4 +loss_x_j5\
                   + 10 * loss_t + loss_a
            loss.backward()

            optimizer.step()

            # update recorder
            train_loss_record.update(loss.item(), batch_size)

            loss_x_jf_record.update(loss_x_jf.item(), batch_size)
            loss_x_j0_record.update(loss_x_j0.item(), batch_size)
            loss_x_j1_record.update(loss_x_j1.item(), batch_size)
            loss_x_j2_record.update(loss_x_j2.item(), batch_size)
            loss_x_j3_record.update(loss_x_j3.item(), batch_size)
            loss_x_j4_record.update(loss_x_j4.item(), batch_size)
            loss_x_j5_record.update(loss_x_j5.item(), batch_size)
            

            loss_t_record.update(loss_t.item(), batch_size)
            loss_a_record.update(loss_a.item(), batch_size)

            curr_iter += 1

            # log = '[iter %d], [train loss %.5f], [loss_x_fusion %.5f], [loss_x_phy %.5f], [loss_x_j1 %.5f], ' \
            #       '[loss_x_j2 %.5f], [loss_x_j3 %.5f], [loss_x_j4 %.5f], [loss_t %.5f], [loss_a %.5f], ' \
            #       '[lr %.13f]' % \
            #       (curr_iter, train_loss_record.avg, loss_x_jf_record.avg, loss_x_j0_record.avg,
            #        loss_x_j1_record.avg, loss_x_j2_record.avg, loss_x_j3_record.avg, loss_x_j4_record.avg,
            #        loss_t_record.avg, loss_a_record.avg, optimizer.param_groups[1]['lr'])
            
            log = '[iter %d], [train loss %.5f], [loss_x_fusion %.5f], [loss_x_phy %.5f], [loss_x_j1 %.5f], ' \
                  '[loss_x_j2 %.5f], [loss_x_j3 %.5f], [loss_x_j4 %.5f],[loss_x_j5 %.5f], [loss_t %.5f], [loss_a %.5f], ' \
                  '[lr %.13f]' % \
                  (curr_iter, train_loss_record.avg, loss_x_jf_record.avg, loss_x_j0_record.avg,
                   loss_x_j1_record.avg, loss_x_j2_record.avg, loss_x_j3_record.avg, loss_x_j4_record.avg,
                   loss_x_j5_record.avg,
                   loss_t_record.avg, loss_a_record.avg, optimizer.param_groups[1]['lr'])
            print(log)
            open(log_path, 'a').write(log + '\n')

            if (curr_iter + 1) % cfgs['val_freq'] == 0:
                validate(net, curr_iter, optimizer)

            if curr_iter > cfgs['iter_num']:
                break


def validate(net, curr_iter, optimizer):
    print('validating...')
    net.eval()

    loss_record = AvgMeter()
    psnr_record, ssim_record, mse_record = AvgMeter(), AvgMeter(), AvgMeter()

    with torch.no_grad():
        for data in tqdm(val_loader):
            haze, gt, _ = data

            haze = haze.cuda()
            gt = gt.cuda()

            dehaze = net(haze)

            loss = criterion(dehaze, gt)
            loss_record.update(loss.item(), haze.size(0))
            for i in range(len(haze)):
                r = dehaze[i].cpu().numpy().transpose([1, 2, 0])  # data range [0, 1]
                g = gt[i].cpu().numpy().transpose([1, 2, 0])
                psnr = peak_signal_noise_ratio(g, r)
                ssim = structural_similarity(g, r, data_range=1, multichannel=True,
                                             gaussian_weights=True, sigma=1.5, use_sample_covariance=False, win_size = 11, channel_axis=2)
                mse = mean_squared_error(g, r)
                psnr_record.update(psnr)
                ssim_record.update(ssim)
                mse_record.update(mse)

    snapshot_name = 'iter_%d_loss_%.5f_lr_%.6f' % (curr_iter + 1, loss_record.avg, optimizer.param_groups[1]['lr'])
    print('[validate]: [iter {}], [loss {:.5f}] [PSNR {:.4f}] [SSIM {:.4f}] [MSE {:.4f}]'.format(
        curr_iter + 1, loss_record.avg, psnr_record.avg, ssim_record.avg, mse_record.avg))
    log = '[validate]: [iter {}], [loss {:.5f}] [PSNR {:.4f}] [SSIM {:.4f}] [MSE {:.4f}]'.format(
        curr_iter + 1, loss_record.avg, psnr_record.avg, ssim_record.avg, mse_record.avg)
    open(log_path, 'a').write(log + '\n')

    if (curr_iter+1>5000):
        torch.save(net.state_dict(),
                os.path.join(args.ckpt_path, args.exp_name, snapshot_name + '.pth'))
        torch.save(optimizer.state_dict(),
                os.path.join(args.ckpt_path, args.exp_name, snapshot_name + '_optim.pth'))

    net.train()


if __name__ == '__main__':
    args = parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    # 启用 cuDNN 的自动优化
    torch.backends.cudnn.benchmark = True

    # 设置当前使用的 GPU 设备
    # 假设 args.gpus 是一个逗号分隔的字符串，例如 "0,1"
    gpus = [int(gpu) for gpu in args.gpus.split(',')]
    if len(gpus) > 0:
        torch.cuda.set_device(gpus[0])
    
    train_dataset = ItsDataset(TRAIN_ITS_ROOT, True, cfgs['crop_size'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfgs['train_batch_size'], num_workers=4,
                              shuffle=True, drop_last=True)

    val_dataset = SotsDataset(TEST_SOTS_ROOT,'val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4)

    criterion = nn.L1Loss().cuda()
    log_path = os.path.join(args.ckpt_path, args.exp_name, str(datetime.datetime.now()) + '.txt')

    main()
