import torch 
from copy import deepcopy
from collections import OrderedDict 
from torch.nn.parallel import DataParallel, DistributedDataParallel
from basicsr.archs.safmn_arch import SAFMN

def get_bare_model(net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net


def convert_safmn_model(net, pretrained_model_path, param_key=['params', 'params_ema']):
    
    ori_net = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)

    save_dict = {}

    for net_, param_key_ in zip(ori_net, param_key):
        crt_net = net.state_dict()
        for ori_k, _ in net_.items():
            if 'att.w_convs' in ori_k:
                crt_k = ori_k.replace('att.w_convs', 'safm.mfr')
            elif 'att.aggr' in ori_k:
                crt_k = ori_k.replace('att.aggr', 'safm.aggr')
            elif 'ffn.fmbconv' in ori_k:
                crt_k = ori_k.replace('ffn.fmbconv', 'ccm.ccm')
            else:
                crt_k = ori_k
            crt_net[crt_k] = ori_net[ori_k]
        save_dict[param_key_] =  crt_net

    torch.save(save_dict, 'experiments/pretrain_model/SAFMN_DF2K_x4_official.pth')

if __name__ == '__main__':
    load_path = 'experiments/SAFMN/MPSR_b64c36n8_500K_DF2K_x4_L1_0.05FFT/models/net_g_450000.pth'
    model = SAFMN(dim=36, n_blocks=8, ffn_scale=2.0, upscaling_factor=4)
    convert_safmn_model(model, load_path)

