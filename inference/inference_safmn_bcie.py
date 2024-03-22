import argparse
import cv2
import glob
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.safmn_bcie_arch import SAFMN_BCIE


# self-ensemble
def test_selfensemble(lq, model):
    def _transform(v, op):
        v2np = v.data.cpu().numpy()
        if op == 'v':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'h':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 't':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()

        ret = torch.Tensor(tfnp).to(lq.device)
        return ret

    # prepare augmented data
    lq_list = [lq]
    for tf in 'v', 'h', 't':
        lq_list.extend([_transform(t, tf) for t in lq_list])

    # inference
    with torch.no_grad():
        out_list = [model(aug) for aug in lq_list]

    # merge results
    for i in range(len(out_list)):
        if i > 3:
            out_list[i] = _transform(out_list[i], 't')
        if i % 4 > 1:
            out_list[i] = _transform(out_list[i], 'h')
        if (i % 4) % 2 == 1:
            out_list[i] = _transform(out_list[i], 'v')
    output = torch.cat(out_list, dim=0)

    return output.mean(dim=0, keepdim=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='datasets/DIV2K/validation_JPEG', help='input test image folder')
    parser.add_argument('--output', type=str, default='results/SAFMN_BCIE/test_results_ensemble', help='output folder')
    parser.add_argument('--model_path', type=str, default='https://github.com/sunny2109/SAFMN/releases/download/v0.1.1/SAFMN_BCIE.pth')
    parser.add_argument('--self_ensemble', action='store_true', help='Using self-ensemble strategy')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up model
    model = SAFMN_BCIE(dim=128, n_blocks=8, num_layers=6, ffn_scale=2.0, upscaling_factor=2)

    if args.model_path.startswith('https://'):
        args.model_path = load_file_from_url(url=args.model_path, model_dir=os.path.join('experiments/pretrained_models'), progress=True, file_name=None)
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)

    model.eval()
    model = model.to(device)

    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
        # read image
        imgname = os.path.splitext(os.path.basename(path))[0]
        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        # img2tensor
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)

        print(f'Processing......idx: {idx}, img: {imgname}, input_size: {img.size()}')

        # inference
        if args.self_ensemble:
            output = test_selfensemble(img, model)
        else:
            with torch.no_grad():
                output = model(img)

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        cv2.imwrite(os.path.join(args.output, f'{imgname}.png'), output)


if __name__ == '__main__':
    main()
