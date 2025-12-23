import os.path
import logging
import torch
import argparse
import json
import glob
import pyiqa
import numpy as np

from PIL import Image
from pprint import pprint
from fvcore.nn import FlopCountAnalysis
from utils.model_summary import get_model_activation
from utils import utils_logger
from utils import utils_image as util

from torchvision import transforms

def select_model(args, device):
    # Model ID is assigned according to the order of the submissions.
    # Different networks are trained with input range of either [0,1] or [0,255]. The range is determined manually.
    model_id = args.model_id

    if model_id == 0:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        name, data_range = f"{model_id:02}_ESRGAN", 1.0
        model_path = os.path.join('model_zoo', 'esrgan.pth')
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        state_dict = torch.load(model_path)
        if 'params_ema' in state_dict:
            model.load_state_dict(state_dict['params_ema'], strict=True)
        else:
            model.load_state_dict(state_dict, strict=True)

    elif model_id == 1:
        from models.span import SPAN30
        name, data_range = f"{model_id:02}_SPAN", 1.0
        model_path = os.path.join('model_zoo', f'span.pth')
        model =SPAN30(3, 3, upscale=4, feature_channels=28)
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict, strict=False)

    elif model_id == 2:
        from models.rrdbnet import RRDBNet
        name, data_range = f"{model_id:02}_BSRGAN", 1.0
        model_path = os.path.join('model_zoo', 'bsrgan.pth')
        model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4)
        state_dict = torch.load(model_path)
        if 'params_ema' in state_dict:
            model.load_state_dict(state_dict['params_ema'], strict=True)
        else:
            model.load_state_dict(state_dict, strict=True)

    elif model_id == 3:
        from models.safmn import SAFMN
        name, data_range = f"{model_id:02}_SAFMN", 1.0
        model_path = os.path.join('model_zoo', 'SAFMN-L.pth')
        model = SAFMN(dim=96, n_blocks=16, ffn_scale=2.0, upscaling_factor=4)
        state_dict = torch.load(model_path)
        if 'params_ema' in state_dict:
            model.load_state_dict(state_dict['params_ema'], strict=True)
        else:
            model.load_state_dict(state_dict, strict=True)

    else:
        raise NotImplementedError(f"Model {model_id} is not implemented.")

    model.eval()
    tile = None
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    return model, name, data_range, tile


def forward(img_lq, model, tile=None, tile_overlap=32, scale=4):
    if tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(tile, h, w)
        tile_overlap = tile_overlap
        sf = scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output


def run(model, model_id, model_name, data_range, tile, logger, device, args, mode="test"):

    sf = 4
    border = sf
    results = dict()
    results[f"{mode}_runtime"] = []
    results[f"{mode}_pi"] = []
    results[f"{mode}_clipiqa"] = []
    results[f"{mode}_maniqa"] = []

    # --------------------------------
    # dataset path
    # --------------------------------

    img_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    img_files = [
    os.path.join(args.data_dir, fname)
    for fname in os.listdir(args.data_dir)
    if fname.lower().endswith(img_extensions)
]
    save_path = os.path.join(args.save_dir, model_name, mode)
    util.mkdir(save_path)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # --------------------------------
    # Out of the loop we initialize the NR-IQA models
    # --------------------------------
    niqe_model = pyiqa.create_metric('niqe').eval().cuda()
    ma_model = pyiqa.create_metric('nrqm').eval().cuda()
    clipiqa_model = pyiqa.create_metric('clipiqa').eval().cuda()
    maniqa_model = pyiqa.create_metric('maniqa').eval().cuda()
 
    for img_path in img_files:
        # --------------------------------
        # (1) img_lr
        # --------------------------------
        img_name, ext = os.path.splitext(os.path.basename(img_path))
        img_lr = util.imread_uint(img_path, n_channels=3)
        img_lr = util.uint2tensor4(img_lr, data_range)
        img_lr = img_lr.to(device)

        # --------------------------------
        # (2) img_sr
        # --------------------------------
        start.record()
        img_sr = forward(img_lr, model, tile)
        end.record()
        torch.cuda.synchronize()
        img_sr = torch.clamp(img_sr, 0.0, 1.0)

        results[f"{mode}_runtime"].append(start.elapsed_time(end))  # milliseconds

        # --------------------------------
        # (3) save img_sr
        # --------------------------------
        output_pil = transforms.ToPILImage()(img_sr[0].cpu())
        outf = os.path.join(save_path, os.path.basename(img_path))
        output_pil.save(outf)

        # --------------------------------
        # PI, ClipIQA, MANIQA
        # --------------------------------
        niqe =  niqe_model(img_sr).item()
        ma = ma_model(img_sr).item()
        pi = 0.5 * ((10-ma)+niqe)
        results[f"{mode}_pi"].append(pi)

        clipiqa = clipiqa_model(img_sr).item()
        results[f"{mode}_clipiqa"].append(clipiqa)

        maniqa = maniqa_model(img_sr).item()
        results[f"{mode}_maniqa"].append(maniqa)


        logger.info("{:s} - PI: {:.2f}; ClipIQA: {:.4f}, MANIQA: {:.4f}.".format(img_name + ext, pi, clipiqa, maniqa))

    results[f"{mode}_memory"] = torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024 ** 2
    results[f"{mode}_ave_runtime"] = sum(results[f"{mode}_runtime"]) / len(results[f"{mode}_runtime"]) 
    results[f"{mode}_ave_pi"] = sum(results[f"{mode}_pi"]) / len(results[f"{mode}_pi"])
    results[f"{mode}_ave_clipiqa"] = sum(results[f"{mode}_clipiqa"]) / len(results[f"{mode}_clipiqa"])
    results[f"{mode}_ave_maniqa"] = sum(results[f"{mode}_maniqa"]) / len(results[f"{mode}_maniqa"])
    logger.info("{:>16s} : {:<.3f} [M]".format("Max Memory", results[f"{mode}_memory"])) 
    logger.info("------> Average runtime of ({}) is : {:.6f} milliseconds".format("test" if mode == "test" else "valid", results[f"{mode}_ave_runtime"]))
    logger.info("------> Average PI of ({}) is : {:.6f} dB".format("test" if mode == "test" else "valid", results[f"{mode}_ave_pi"]))
    logger.info("------> Average ClipIQA of ({}) is : {:.6f} dB".format("test" if mode == "test" else "valid", results[f"{mode}_ave_clipiqa"]))
    logger.info("------> Average MANIQA of ({}) is : {:.6f} dB".format("test" if mode == "test" else "valid", results[f"{mode}_ave_maniqa"]))

    return results


def main(args):

    utils_logger.logger_info("CVPR2025-RT4KPSR", log_path="CVPR2025-RT4KPSR.log")
    logger = logging.getLogger("CVPR2025-RT4KPSR")

    # --------------------------------
    # basic settings
    # --------------------------------
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    json_dir = os.path.join(os.getcwd(), "results.json")
    if not os.path.exists(json_dir):
        results = dict()
    else:
        with open(json_dir, "r") as f:
            results = json.load(f)

    # --------------------------------
    # load model
    # --------------------------------
    model, model_name, data_range, tile = select_model(args, device)
    logger.info(model_name)

    # if model not in results:
    if True:
        # --------------------------------
        # restore image
        # --------------------------------

        # inference on the DIV2K_LSDIR_valid set
        valid_results = run(model, args.model_id, model_name, data_range, tile, logger, device, args, mode="valid")
        # record PSNR, runtime
        results[model_name] = valid_results

        # inference conducted by the Organizer on DIV2K_LSDIR_test set
        if args.include_test:
            test_results = run(model, model_name, data_range, tile, logger, device, args, mode="test")
            results[model_name].update(test_results)

        input_dim = (3, 960, 540)  # set the input dimension
        activations, num_conv = get_model_activation(model, input_dim)
        activations = activations/10**6
        logger.info("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
        logger.info("{:>16s} : {:<d}".format("#Conv2d", num_conv))

        # The FLOPs calculation in previous NTIRE_ESR Challenge
        # flops = get_model_flops(model, input_dim, False)
        # flops = flops/10**9
        # logger.info("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

        # fvcore is used for FLOPs calculation
        input_fake = torch.rand(1, 3, 960, 540).to(device)
        flops = FlopCountAnalysis(model, input_fake).total()
        flops = flops/10**9
        logger.info("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

        num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
        num_parameters = num_parameters/10**6
        logger.info("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
        results[model_name].update({"activations": activations, "num_conv": num_conv, "flops": flops, "num_parameters": num_parameters})

        with open(json_dir, "w") as f:
            json.dump(results, f)
    if args.include_test:
        fmt = "{:20s}\t{:10s}\t{:10s}\t{:14s}\t{:14s}\t{:14s}\t{:10s}\t{:10s}\t{:8s}\t{:8s}\t{:8s}\n"
        s = fmt.format("Model", "Val PI", "Test PI", "Val ClipIQA", "Test ClipIQA", "Val MANIQA", "Test MANIQA", "Val Time [ms]", "Test Time [ms]", "Ave Time [ms]",
                       "Params [M]", "FLOPs [G]", "Acts [M]", "Mem [M]", "Conv")
    else:
        fmt = "{:20s}\t{:10s}\t{:14s}\t{:10s}\t{:10s}\t{:8s}\t{:8s}\t{:8s}\n"
        s = fmt.format("Model", "Val PI", "Val ClipIQA", "Val MANIQA", "Val Time [ms]", "Params [M]", "FLOPs [G]", "Acts [M]", "Mem [M]", "Conv")
    for k, v in results.items():
        val_pi = f"{v['valid_ave_pi']:2.4f}"
        val_clipiqa = f"{v['valid_ave_clipiqa']:2.4f}"
        val_maniqa = f"{v['valid_ave_maniqa']:2.4f}"
        val_time = f"{v['valid_ave_runtime']:3.4f}"
        mem = f"{v['valid_memory']:2.2f}"
        
        num_param = f"{v['num_parameters']:2.3f}"
        flops = f"{v['flops']:2.2f}"
        acts = f"{v['activations']:2.2f}"
        conv = f"{v['num_conv']:4d}"
        if args.include_test:
            # from IPython import embed; embed()
            test_pi = f"{v['test_ave_pi']:2.4f}"
            test_clipiqa = f"{v['test_ave_clipiqa']:2.4f}"
            test_maniqa = f"{v['test_ave_maniqa']:2.4f}"
            test_time = f"{v['test_ave_runtime']:3.4f}"
            ave_time = f"{(v['valid_ave_runtime'] + v['test_ave_runtime']) / 2:3.2f}"
            s += fmt.format(k, val_pi, test_pi, val_clipiqa, test_clipiqa, val_maniqa, test_maniqa, val_time, test_time, ave_time, num_param, flops, acts, mem, conv)
        else:
            s += fmt.format(k, val_pi, val_clipiqa, val_maniqa, val_time, num_param, flops, acts, mem, conv)
    with open(os.path.join(os.getcwd(), 'results.txt'), "w") as f:
        f.write(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CVPR2025-RT4KPSR")
    parser.add_argument("--data_dir", default="../", type=str)
    parser.add_argument("--save_dir", default="../results", type=str)
    parser.add_argument("--model_id", default=0, type=int)
    parser.add_argument("--include_test", action="store_true", help="Inference on the DIV2K_LSDIR test set")

    args = parser.parse_args()
    pprint(args)

    main(args)