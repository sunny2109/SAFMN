import os
import torch
import cv2
import argparse
import pillow_avif
import numpy as np

from light_safmnpp_arch import SAFMN_VIS24
from PIL import Image


def aviffrombytes(path, float32=False):
    avifimg = Image.open(path).convert("RGB")
    img = np.array(avifimg)

    if float32:
        img = img.astype(np.float32) / 255.
    return img

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)
    
def save_rgb (img, filename):
    '''Save RGB image <img> as 8bit 3-channel using the provided <filename>'''
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    if np.max(img) <= 1:
        img = img * 255
    
    img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./test_data', help='input test image folder')
    parser.add_argument('--output', type=str, default='./results', help='output folder')
    parser.add_argument('--model_path', type=str, default='./pretrained_model/light_safmnpp.pth')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SAFMN_VIS24(dim=32, n_blocks=2, ffn_scale=1.5).to(device)
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()

    for k, v in model.named_parameters():
        v.requires_grad = False

    for img in os.listdir(args.input):
        basename, ext = os.path.splitext(os.path.basename(img))
        print(basename)
        
        lq_path = os.path.join(args.input, img)
        
        result_path = os.path.join(args.output, f"{basename}.png")

        lq_img = aviffrombytes(lq_path, float32=True)
        lq_img = img2tensor(lq_img, bgr2rgb=False, float32=True).to(device)

        result = model(lq_img.unsqueeze(0))
        save_rgb(result[0], result_path)


if __name__ =="__main__":
    main()