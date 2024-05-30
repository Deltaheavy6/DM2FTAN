# coding: utf-8
import os

import numpy as np
import torch
from torch import nn
from torchvision import transforms

from tools.config import TEST_SOTS_ROOT, OHAZE_ROOT,HazeRD_ROOT
from tools.utils import AvgMeter, check_mkdir, sliding_forward
from model import DM2FNet,DM2FNetTAN,MAmba2Net,DM2FNetTAN2
from datasets import SotsDataset, OHazeDataset
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from sklearn.metrics import mean_squared_error
from skimage.color import deltaE_ciede2000, rgb2lab
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.manual_seed(2018)
torch.cuda.set_device(0)

ckpt_path = './ckpt/tan'
# exp_name = 'RESIDE_ITS'
exp_name = 'tan_exp3'

args = {
    
    # 'snapshot': 'iter_21000_loss_0.01304_lr_0.000380',
    'snapshot': 'iter_10000_loss_0.01649_lr_0.000443',
    # 'snapshot':'iter_69000_loss_0.01041_lr_0.000084'
}

to_test = {
    # 'SOTS': TEST_SOTS_ROOT,
    # 'O-Haze': OHAZE_ROOT,
    'HazeRD':HazeRD_ROOT
}

to_pil = transforms.ToPILImage()



def main():
    with torch.no_grad():
        criterion = nn.L1Loss().cuda()

        for name, root in to_test.items():
            if 'SOTS' in name:
                net = DM2FNetTAN().cuda()
                dataset = SotsDataset(root)
            elif 'O-Haze' in name:
                net = DM2FNetTAN().cuda()
                dataset = OHazeDataset(root, 'test')
            elif 'HazeRD' in name:
                net = DM2FNetTAN().cuda()
                dataset = OHazeDataset(root,'HazeRD')
            else:
                raise NotImplementedError

            # net = nn.DataParallel(net)

            if len(args['snapshot']) > 0:
                print('load snapshot \'%s\' for testing' % args['snapshot'])
                net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))

            net.eval()
            dataloader = DataLoader(dataset, batch_size=1)

            psnrs, ssims = [], []
            loss_record = AvgMeter()
            mse_scores = []
            ciede2000_scores = []

            for idx, data in enumerate(dataloader):
                # haze_image, _, _, _, fs = data
                haze, gts, fs = data
                # print(haze.shape, gts.shape)

                check_mkdir(os.path.join(ckpt_path, exp_name,
                                         '(%s) %s_%s' % (exp_name, name, args['snapshot'])))

                haze = haze.cuda()

                if 'O-Haze' in name:
                    res = sliding_forward(net, haze).detach()
                else:
                    res = net(haze).detach()

                loss = criterion(res, gts.cuda())
                loss_record.update(loss.item(), haze.size(0))
                

                for i in range(len(fs)):
                    r = res[i].cpu().numpy().transpose([1, 2, 0])
                    gt = gts[i].cpu().numpy().transpose([1, 2, 0])
                    # 计算 MSE
                    mse = mean_squared_error(gt.reshape(-1, 3), r.reshape(-1, 3))
                    mse_scores.append(mse)

                    
                    psnr = peak_signal_noise_ratio(gt, r)
                    psnrs.append(psnr)
                    ssim = structural_similarity(gt, r, win_size = 3,data_range=1, multichannel=True,
                                                 gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                                                 channel_axis=-1)
                    ssims.append(ssim)
                    
                    # 计算 CIEDE2000
                    gt_lab = rgb2lab(gt)
                    r_lab = rgb2lab(r)
                    # gt_lab = convert_color(gt, LabColor)
                    # r_lab = convert_color(r, LabColor)
                    # gt_lab = XYZ_to_Lab(gt_xyz)
                    # r_lab = XYZ_to_Lab(r_xyz)
                    # print(gt_lab.__class__.__name__)
                    ciede2000 = deltaE_ciede2000(gt_lab, r_lab)
                    ciede2000_avg = ciede2000.mean()
                    ciede2000_scores.append(ciede2000.mean())
                    print('predicting for {} ({}/{}) [{}]: MSE: {:.4f}, PSNR {:.4f}, SSIM {:.4f}, CIEDE2000: {:.4f}'
                          .format(name, idx + 1, len(dataloader), fs[i], mse, psnr, ssim, ciede2000_avg))

                for r, f in zip(res.cpu(), fs):
                    to_pil(r).save(
                        os.path.join(ckpt_path, exp_name,
                                     '(%s) %s_%s' % (exp_name, name, args['snapshot']), '%s.png' % f))

            print(f"[{name}] L1: {loss_record.avg:.6f}, MSE: {np.mean(mse_scores):.6f}, PSNR: {np.mean(psnrs):.6f}, SSIM: {np.mean(ssims):.6f}, CIEDE2000: {np.mean(ciede2000_scores):.6f}")



if __name__ == '__main__':
    main()
