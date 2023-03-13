import cv2
import glob
import logging
import numpy as np
import os.path as osp
from torchvision.transforms.functional import normalize
from basicsr.utils import img2tensor
from basicsr.utils import get_root_logger, get_time_str

try:
    import lpips
except ImportError:
    print('Please install lpips: pip install lpips')


def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img


def main():
    # Configurations
    log_file = osp.join('results', f"MPSR_Set14_LPIPS_x3.log")
    logger = get_root_logger(logger_name='Benchmarks', log_level=logging.INFO, log_file=log_file)
    # -------------------------------------------------------------------------
    folder_gt = '/media/sunlong/78C88475C8843402/Codes/BasicSR_torch1.8/datasets/Benchmarks/Set14/HR'
    folder_restored = 'results/MPSR_b64c36n8_500K_DF2K_x3_L1_0.05FFT/visualization/Set14/'
    # crop_border = 4
    suffix = 'x3_MPSR_b64c36n8_500K_DF2K_x3_L1_0.05FFT'
    # -------------------------------------------------------------------------
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()  # RGB, normalized to [-1,1]
    lpips_all = []
    img_list = sorted(glob.glob(osp.join(folder_gt, '*')))

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    for i, img_path in enumerate(img_list):
        basename, ext = osp.splitext(osp.basename(img_path))
        print(basename)
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        img_restored = cv2.imread(osp.join(folder_restored, basename + suffix + ext), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        img_gt = modcrop(img_gt, 3)
        img_restored = modcrop(img_restored, 3)

        img_gt, img_restored = img2tensor([img_gt, img_restored], bgr2rgb=True, float32=True)
        # norm to [-1, 1]
        normalize(img_gt, mean, std, inplace=True)
        normalize(img_restored, mean, std, inplace=True)

        # calculate lpips
        lpips_val = loss_fn_vgg(img_restored.unsqueeze(0).cuda(), img_gt.unsqueeze(0).cuda())

        # print(f'{i+1:3d}: {basename:25}. \tLPIPS: {lpips_val.item():.6f}.')
        logger.info(f'{i+1:3d}: {basename:25}. \tLPIPS: {lpips_val.item():.6f}.')
        lpips_all.append(lpips_val)

    # print(f'Average: LPIPS: {sum(lpips_all).item() / len(lpips_all):.6f}')
    logger.info(f'Average: LPIPS: {sum(lpips_all).item() / len(lpips_all):.6f}')


if __name__ == '__main__':
    main()
