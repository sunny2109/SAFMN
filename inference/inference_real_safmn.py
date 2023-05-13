# Modified from https://github.com/megvii-research/TLC/blob/main/basicsr/models/archs/restormer_arch.py
import argparse
import cv2
import glob
import numpy as np
import os
import torch
from torch.nn import functional as F
from basicsr.archs.safmn_arch import SAFMN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='datasets/test_demo', help='input test image folder')
    parser.add_argument('--output', type=str, default='results/SAFMN/test_demo', help='output folder')
    parser.add_argument('--large_input', action='store_true', help='the input image with large resolution, we crop the input into sub-images for memory-efficient forward')
    parser.add_argument('--model_path', type=str, default='experiments/pretrain_model/SAFMN_L_Real_LSDIR_x4.pth')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = SAFMN(dim=128, n_blocks=16, ffn_scale=2.0, upscaling_factor=4)
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)

    model.eval()
    model = model.to(device)

    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
        # read image
        imgname = os.path.splitext(os.path.basename(path))[0]
        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)

        print(f'Testing......idx: {idx}, img: {imgname}, input_size: {img.size()}')

        # inference
        if args.large_input:
            img, idx, size = img2patch(img)
            
            with torch.no_grad():
                n = len(img)
                outs = []
                m = 1
                i = 0
                while i < n:
                    j = i + m
                    if j >= n:
                        j = n
                    pred = output = model(img[i:j])
                    if isinstance(pred, list):
                        pred = pred[-1]
                    outs.append(pred.detach())
                    i = j
                output = torch.cat(outs, dim=0)
                print(f'Max Memery [M]: {torch.cuda.max_memory_allocated(torch.cuda.current_device())/1024**2}')

            output = patch2img(output, idx, size)
        else:
            with torch.no_grad():
                output = model(img)
                print(f'Max Memery [M]: {torch.cuda.max_memory_allocated(torch.cuda.current_device())/1024**2}')

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        cv2.imwrite(os.path.join(args.output, f'{imgname}_p512_SAFMN.png'), output)


def img2patch(lq, scale=4, crop_size=512):
    b, c, hl, wl = lq.size()    
    h, w = hl*scale, wl*scale
    sr_size = (b, c, h, w)
    assert b == 1

    crop_size_h, crop_size_w = crop_size // scale * scale, crop_size // scale * scale

    #adaptive step_i, step_j
    num_row = (h - 1) // crop_size_h + 1
    num_col = (w - 1) // crop_size_w + 1

    import math
    step_j = crop_size_w if num_col == 1 else math.ceil((w - crop_size_w) / (num_col - 1) - 1e-8)
    step_i = crop_size_h if num_row == 1 else math.ceil((h - crop_size_h) / (num_row - 1) - 1e-8)

    step_i = step_i // scale * scale
    step_j = step_j // scale * scale

    parts = []
    idxes = []

    i = 0  # 0~h-1
    last_i = False
    while i < h and not last_i:
        j = 0
        if i + crop_size_h >= h:
            i = h - crop_size_h
            last_i = True

        last_j = False
        while j < w and not last_j:
            if j + crop_size_w >= w:
                j = w - crop_size_w
                last_j = True
            parts.append(lq[:, :, i // scale :(i + crop_size_h) // scale, j // scale:(j + crop_size_w) // scale])
            idxes.append({'i': i, 'j': j})
            j = j + step_j
        i = i + step_i

    return torch.cat(parts, dim=0), idxes, sr_size


def patch2img(outs, idxes, sr_size, scale=4, crop_size=512):
    preds = torch.zeros(sr_size).to(outs.device)
    b, c, h, w = sr_size

    count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
    crop_size_h, crop_size_w = crop_size // scale * scale, crop_size // scale * scale

    for cnt, each_idx in enumerate(idxes):
        i = each_idx['i']
        j = each_idx['j']
        preds[0, :, i: i + crop_size_h, j: j + crop_size_w] += outs[cnt]
        count_mt[0, 0, i: i + crop_size_h, j: j + crop_size_w] += 1.

    return (preds / count_mt).to(outs.device)


if __name__ == '__main__':
    main()