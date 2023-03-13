import argparse
import cv2
import os
from os import path as osp
import warnings
import logging
from basicsr.utils import scandir
from basicsr.metrics import calculate_niqe
from basicsr.utils.matlab_functions import bgr2ycbcr
from basicsr.utils import get_root_logger, get_time_str


def main(args):
    log_file = osp.join('results', f"MPSR_Set5_NIQE_x4.log")
    logger = get_root_logger(logger_name='Benchmarks', log_level=logging.INFO, log_file=log_file)
    niqe_all = []
    img_list = sorted(scandir(args.input, recursive=True, full_path=True))

    for i, img_path in enumerate(img_list):
        basename, _ = os.path.splitext(os.path.basename(img_path))
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            niqe_score = calculate_niqe(img, args.crop_border, input_order='HWC', convert_to='y')
        # print(f'{i+1:3d}: {basename:25}. \tNIQE: {niqe_score:.6f}')
        logger.info(f'{i+1:3d}: {basename:25}. \tNIQE: {niqe_score:.6f}')
        niqe_all.append(niqe_score)

    # print(args.input)
    logger.info(args.input)
    # print(f'Average: NIQE: {sum(niqe_all) / len(niqe_all):.6f}')
    logger.info(f'Average: NIQE: {sum(niqe_all) / len(niqe_all):.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='results/MPSR_b64c36n8_500K_DF2K_x4_L1_0.05FFT/visualization/Set5', help='Input path')
    parser.add_argument('--crop_border', type=int, default=0, help='Crop border for each side')
    args = parser.parse_args()
    main(args)
