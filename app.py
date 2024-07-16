import sys
sys.path.append('SAFMN')

import os
import cv2
import argparse
import glob
import numpy as np
import os
import torch
import torch.nn.functional as F
import gradio as gr

from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.colorfix import wavelet_reconstruction
from basicsr.archs.safmn_arch import SAFMN


pretrain_model_url = {
	'safmn_x2': 'https://github.com/sunny2109/SAFMN/releases/download/v0.1.0/SAFMN_L_Real_LSDIR_x2-v2.pth',
	'safmn_x4': 'https://github.com/sunny2109/SAFMN/releases/download/v0.1.0/SAFMN_L_Real_LSDIR_x4-v2.pth',
}


# download weights
if not os.path.exists('./experiments/pretrained_models/SAFMN_L_Real_LSDIR_x2-v2.pth'):
	load_file_from_url(url=pretrain_model_url['safmn_x2'], model_dir='./experiments/pretrained_models/', progress=True, file_name=None)

if not os.path.exists('./experiments/pretrained_models/SAFMN_L_Real_LSDIR_x4-v2.pth'):
	load_file_from_url(url=pretrain_model_url['safmn_x4'], model_dir='./experiments/pretrained_models/', progress=True, file_name=None)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_safmn(upscale):
	model = SAFMN(dim=128, n_blocks=16, ffn_scale=2.0, upscaling_factor=upscale)
	if upscale == 2:
		model_path = './experiments/pretrained_models/SAFMN_L_Real_LSDIR_x2-v2.pth'
	elif upscale == 4:
		model_path = './experiments/pretrained_models/SAFMN_L_Real_LSDIR_x4-v2.pth'
	else:
		raise NotImplementedError('Only support x2/x4 upscaling!')

	model.load_state_dict(torch.load(model_path)['params'], strict=True)
	model.eval()
	return model.to(device)


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


os.makedirs('./results', exist_ok=True)

def inference(image, upscale, large_input_flag, color_fix):
	upscale = int(upscale) # convert type to int
	if upscale > 4: 
		upscale = 4 
	if 0 < upscale < 3:
		upscale = 2

	model = set_safmn(upscale)

	img = cv2.imread(str(image), cv2.IMREAD_COLOR)
	print(f'input size: {img.shape}')

	# img2tensor
	img = img.astype(np.float32) / 255.
	img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
	img = img.unsqueeze(0).to(device)

	# inference
	if large_input_flag:
		patches, idx, size = img2patch(img, scale=upscale)
		with torch.no_grad():
			n = len(patches)
			outs = []
			m = 1
			i = 0
			while i < n:
				j = i + m
				if j >= n:
					j = n
				pred = output = model(patches[i:j])
				if isinstance(pred, list):
					pred = pred[-1]
				outs.append(pred.detach())
				i = j
			output = torch.cat(outs, dim=0)

		output = patch2img(output, idx, size, scale=upscale)
	else:
		with torch.no_grad():
			output = model(img)

	# color fix
	if color_fix:
		img = F.interpolate(img, scale_factor=upscale, mode='bilinear')
		output = wavelet_reconstruction(output, img)
	# tensor2img
	output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
	if output.ndim == 3:
		output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
	output = (output * 255.0).round().astype(np.uint8)

	# save restored img
	save_path = f'results/out.png'
	cv2.imwrite(save_path, output)

	output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
	return output, save_path



title = "Spatially-Adaptive Feature Modulation for Efficient Image Super-Resolution"
description = r"""
<b>Official Gradio demo</b> for <a href='https://github.com/sunny2109/SAFMN' target='_blank'><b>Spatially-Adaptive Feature Modulation for Efficient Image Super-Resolution (ICCV 2023)</b></a>.<br>
"""
article = r"""
If SAFMN is helpful, please help to ⭐ the <a href='https://github.com/sunny2109/SAFMN' target='_blank'>Github Repo</a>. Thanks!
[![GitHub Stars](https://img.shields.io/github/stars/sunny2109/SAFMN?style=social)](https://github.com/sunny2109/SAFMN)

---
📝 **Citation**

If our work is useful for your research, please consider citing:
```bibtex
@inproceedings{sun2023safmn,
    title={Spatially-Adaptive Feature Modulation for Efficient Image Super-Resolution},
    author={Sun, Long and Dong, Jiangxin and Tang, Jinhui and Pan, Jinshan},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    year={2023}
}
```

<center><img src='https://visitor-badge.laobi.icu/badge?page_id=sunny2109/SAFMN' alt='visitors'></center>
"""

demo = gr.Interface(
    inference, [
        gr.inputs.Image(type="filepath", label="Input"),
        gr.inputs.Number(default=2, label="Upscaling factor (up to 4)"),
		gr.inputs.Checkbox(default=False, label="Memory-efficient inference"),
        gr.inputs.Checkbox(default=False, label="Color correction"),
    ], [
        gr.outputs.Image(type="numpy", label="Output"),
        gr.outputs.File(label="Download the output")
    ],
    title=title,
    description=description,
    article=article,       
)

demo.queue(concurrency_count=2)
demo.launch()
