import json
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

from tqdm import tqdm
from metrics.lpipsPyTorch.modules.lpips import LPIPS

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).reshape(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

class Metrics:
    def __init__(self, device) -> None:
        self.lpips = LPIPS().to(device)
        self.metrics_dict = {}

    def clean(self):
        self.metrics_dict = {}

    def compute_metrics(self, pred_rgbs, gt_rgbs):
        assert pred_rgbs.shape == gt_rgbs.shape
        B = pred_rgbs.shape[0]
        metrics={'psnr': [], 'lpips': [], 'ssim': []}
        pred_rgbs = pred_rgbs.permute(0, 3, 1, 2)
        gt_rgbs = gt_rgbs.permute(0, 3, 1, 2)

        for i in range(B):
            pred_rgb = pred_rgbs[i]
            gt_rgb = gt_rgbs[i]

            metrics['ssim'] += [ssim(pred_rgb, gt_rgb).cpu()]
            metrics['lpips'] += [self.lpips(pred_rgb, gt_rgb).reshape(1).cpu()]
            metrics['psnr'] += [psnr(pred_rgb, gt_rgb).mean().cpu()]

        for k in metrics:
            self.metrics_dict[k] = self.metrics_dict.get(k, []) + metrics[k]
        return metrics
    
    def get_accumulated_metrics(self):
        k = list(self.metrics_dict.keys())[0]
        v = self.metrics_dict[k]
        print(len(v), v[0].shape)
        return {k:torch.tensor(v).mean().cpu().item() for k, v in self.metrics_dict.items()}
    


# def run_metrics(output_dir, dataset, pred_file_names):
#     res = dataset.img_res
#     gt_images_paths = dataset.data['image_paths']
#     img_names = dataset.data['img_name']
#     device='cuda:0'
    
#     metric_loggers = {k: Metrics(device) for k in pred_file_names}
#     for i in tqdm(range(len(gt_images_paths))):
#         gt_image = torch.from_numpy(load_rgb(gt_images_paths[i], res)).float().permute(1, 2, 0).to(device)

#         for pred_file_name in pred_file_names:
#             pred_path = os.path.join(output_dir, pred_file_name, f'{img_names[i]}.png')
#             pred_image = torch.from_numpy(load_rgb(pred_path, res)).float().permute(1, 2, 0).to(device)
#             metric_loggers[pred_file_name].compute_metrics(pred_image[None, ...], gt_image[None, ...])

        
#     metrics_dict = {k:logger.get_accumulated_metrics() for k, logger in metric_loggers.items()}
#     with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
#         json.dump(metrics_dict, f, indent=4)

#     for k in metrics_dict:
#         print('\n--------------------')
#         print(k)
#         print(metrics_dict[k])



    
