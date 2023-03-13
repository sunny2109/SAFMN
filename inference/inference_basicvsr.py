import argparse
import cv2
import glob
import os
import shutil
import torch

from basicsr.archs.basicvsr_arch import BasicVSR, IconVSR
from basicsr.data.data_util import read_img_seq
from basicsr.utils.img_util import tensor2img


def inference(imgs, imgnames, model, save_path):
    with torch.no_grad():
        outputs = model(imgs)
    # save imgs
    outputs = [outputs[:, i, ...].squeeze() for i in range(outputs.size(1))]
    # outputs = list(outputs)

    for output, imgname in zip(outputs, imgnames):
        output = tensor2img(output)
        cv2.imwrite(os.path.join(save_path, f'{imgname}_BasicVSR.png'), output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='BasicVSR')
    parser.add_argument('--model_path', type=str, default='experiments/pretrained_models/BasicVSR_REDS.pth')
    parser.add_argument(
        '--input_path', type=str, default='datasets/SPMCs/LR/LRx4', help='input test image folder')
    parser.add_argument('--save_path', type=str, default='results/BasicVSR_SPMCs_inference_20', help='save image path')
    parser.add_argument('--interval', type=int, default=20, help='interval size')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up model
    if args.model_name == 'BasicVSR':
        model = BasicVSR(num_feat=64, num_block=30)
    elif args.model_name == 'IconVSR':
        model = IconVSR(num_feat=64, num_block=30)
    else:
        raise NotImplementedError('Thw model is not implemented!')
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    # extract images from video format files
    input_path = args.input_path
    use_ffmpeg = False
    if not os.path.isdir(input_path):
        use_ffmpeg = True
        video_name = os.path.splitext(os.path.split(args.input_path)[-1])[0]
        input_path = os.path.join('./BasicVSR_tmp', video_name)
        os.makedirs(os.path.join('./BasicVSR_tmp', video_name), exist_ok=True)
        os.system(f'ffmpeg -i {args.input_path} -qscale:v 1 -qmin 1 -qmax 1 -vsync 0  {input_path} /frame%08d.png')

    # load data and inference
    fname = sorted(os.listdir(input_path))
    for fn in fname:
        save_path = os.path.join(args.save_path, fn)
        os.makedirs(save_path, exist_ok=True)
        imgs_list = sorted(glob.glob(os.path.join(input_path, fn, '*')))
        num_imgs = len(imgs_list)
        if len(imgs_list) <= args.interval:  # too many images may cause CUDA out of memory
            imgs, imgnames = read_img_seq(imgs_list, return_imgname=True)
            imgs = imgs.unsqueeze(0).to(device)
            inference(imgs, imgnames, model, save_path)
        else:
            for idx in range(0, num_imgs, args.interval):
                interval = min(args.interval, num_imgs - idx)
                imgs, imgnames = read_img_seq(imgs_list[idx:idx + interval], return_imgname=True)
                imgs = imgs.unsqueeze(0).to(device)
                inference(imgs, imgnames, model, save_path)

    # delete ffmpeg output images
    if use_ffmpeg:
        shutil.rmtree(input_path)


if __name__ == '__main__':
    main()
