import os
import argparse
import numpy as np
import os.path as osp
import logging
import time
import torch

import basicsr.utils.img_util as util
from basicsr.archs.safmn_arch import SAFMN
from collections import OrderedDict
from basicsr.utils import get_root_logger, get_time_str
from basicsr.utils.model_summary_util import get_model_activation, get_model_flops


'''
This code can help you to calculate:
`FLOPs`, `#Params`, `Runtime`, `#Activations`, `#Conv`, and `Max Memory Allocated`.

- `#Params' denotes the total number of parameters.
- `FLOPs' is the abbreviation for floating point operations.
- `#Activations' measures the number of elements of all outputs of convolutional layers.
- `Memory' represents maximum GPU memory consumption according to the PyTorch function torch.cuda.max_memory_allocated().
- `#Conv' represents the number of convolutional layers.
- `FLOPs', `#Activations', and `Memory' are tested on an LR image of size 256x256.

For more information, please refer to ECCVW paper "AIM 2020 Challenge on Efficient Super-Resolution: Methods and Results".

# If you use this code, please consider the following citations:

@inproceedings{zhang2020aim,
  title={AIM 2020 Challenge on Efficient Super-Resolution: Methods and Results},
  author={Kai Zhang and Martin Danelljan and Yawei Li and Radu Timofte and others},
  booktitle={European Conference on Computer Vision Workshops},
  year={2020}
}
@inproceedings{zhang2019aim,
  title={AIM 2019 Challenge on Constrained Super-Resolution: Methods and Results},
  author={Kai Zhang and Shuhang Gu and Radu Timofte and others},
  booktitle={IEEE International Conference on Computer Vision Workshops},
  year={2019}
}

CuDNN (https://developer.nvidia.com/rdp/cudnn-archive) should be installed.

For `Memery` and `Runtime`, set 'print_modelsummary = False' and 'save_results = False'.
'''


def main(args):
    save_path = osp.join(args.save_path, args.model_name)
    util.mkdir(save_path)
    
    # Set log file
    log_file = osp.join(args.log_path, args.model_name, f'SAFMN_runtime_test_.log')
    logger = get_root_logger(logger_name='Runtime', log_level=logging.INFO, log_file=log_file)

    logger.info(torch.__version__)               # pytorch version
    logger.info(torch.version.cuda)              # cuda version
    logger.info(torch.backends.cudnn.version())  # cudnn version
    logger.info('{:>16s} : {:s}'.format('Model Name', args.model_name))

    torch.cuda.set_device(0)      # set GPU ID
    logger.info('{:>16s} : {:<d}'.format('GPU ID', torch.cuda.current_device()))

    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define network and load model
    model = SAFMN(dim=36, n_blocks=8, ffn_scale=2.0, upscaling_factor=4)
    model.load_state_dict(torch.load(args.pretrain_model)['params'], strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    # print model summary
    if args.print_modelsummary:

        input_dim = (3, 256, 256)  # set the input dimension

        activations, num_conv2d = get_model_activation(model, input_dim)
        logger.info('{:>16s} : {:<.4f} [M]'.format('#Activations', activations/10**6))
        logger.info('{:>16s} : {:<d}'.format('#Conv2d', num_conv2d))

        flops = get_model_flops(model, input_dim, False)
        logger.info('{:>16s} : {:<.4f} [G]'.format('FLOPs', flops/10**9))

        num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
        logger.info('{:>16s} : {:<.4f} [M]'.format('#Params', num_parameters/10**6))

    logger.info('{:>16s} : {:s}'.format('Input Path', args.lr_path))
    logger.info('{:>16s} : {:s}'.format('Output Path', save_path))

    # record runtime
    test_results = OrderedDict()
    test_results['runtime'] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    idx = 0
    for img in util.get_image_paths(args.lr_path):
        idx += 1
        img_name, ext = os.path.splitext(os.path.basename(img))
        logger.info('{:->4d}--> {:>10s}'.format(idx, img_name+ext))

        # Read LR Image
        img_L = util.imread_uint(img, n_channels=3)
        img_L = util.uint2tensor4(img_L)
        torch.cuda.empty_cache()
        img_L = img_L.to(device)

        start.record()
        img_E = model(img_L)
        # logger.info('{:>16s} : {:<.3f} [M]'.format('Max Memery', torch.cuda.max_memory_allocated(torch.cuda.current_device())/1024**2))  # Memery
        end.record()
        torch.cuda.synchronize()
        test_results['runtime'].append(start.elapsed_time(end))  # millisecond

        # get SR image
        img_E = util.tensor2uint(img_E)

        if args.save_results:
            util.imsave(img_E, os.path.join(save_path, img_name+ext))
    ave_runtime = sum(test_results['runtime']) / len(test_results['runtime']) / 1000.0
    logger.info('------> Average runtime of ({}) is : {:.6f} seconds'.format(args.lr_path, ave_runtime))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='SAFMN', help='method name')
    parser.add_argument('--lr_path', type=str, default='datasets/test', help='Path to the LR image')
    parser.add_argument('--log_path', type=str, default='results/', help='Path to log file')
    parser.add_argument('--save_results', action='store_true', help='if true save SR results')
    parser.add_argument('--print_modelsummary', action='store_true', help='if true print modelsummary; set False when calculating `Max Memery` and `Runtime`')
    parser.add_argument('--save_path', type=str, default='results/', help='Path to results')
    parser.add_argument('--pretrain_model', type=str, default='pretrained_model/SAFMN_DF2K_x4.pth', help='Path to the pretrained model')
    args = parser.parse_args()

    main(args)
