import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops
from basicsr.utils.registry import ARCH_REGISTRY


# Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# Convolutional Channel Mixer: Conv3*3 -> Conv1*1
class CCM(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.Conv1 = nn.Conv2d(dim, 2*hidden_dim, 1, 1, 0)
        self.Conv2 = nn.Sequential(nn.Conv2d(2*hidden_dim,2* hidden_dim, (2,5), 1, padding='same',groups=2*hidden_dim),nn.Conv2d(2*hidden_dim,2* hidden_dim, (5,2), 1, padding='same',groups=2*hidden_dim))
        self.Act =nn.GELU()
        self.Conv3 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)


    def forward(self, x):
        x = self.Conv1(x)
        x1,x2 = self.Conv2(x).chunk(2,dim=1)
        x = self.Conv3(x1*x2)
        return x


# Spatially-Adaptive Feature Modulation
class SAFM(nn.Module):
    def __init__(self, dim=60):
        super(SAFM, self).__init__()
        self.dwConv1 = nn.Sequential(nn.Conv2d(dim,dim,3,1,1,groups=dim))
        self.dwConv2 = nn.Sequential(nn.Conv2d(dim//2,dim//2,3,1,1,groups=dim//2))
        self.dwConv3 = nn.Sequential(nn.Conv2d(dim//4,dim//4,3,1,1,groups=dim//4))
        self.ConvCat = nn.Conv2d(dim,dim,1)
        self.Act =nn.GELU()
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x1, x2 = self.Act(self.dwConv1(x)).chunk(2, dim=1)

        x2 = F.adaptive_max_pool2d(x2, (h//8, w//8))  # x8
        x2, x3 = self.Act(self.dwConv2(x2)).chunk(2, dim=1)

        x3 = F.adaptive_max_pool2d(x3, (h//16, w//16)) #x16
        x3 = self.Act(self.dwConv3(x3))

        x2 = F.interpolate(x2, size=(h, w), mode='nearest')
        x3 = F.interpolate(x3, size=(h, w), mode='nearest')
        out = torch.cat([x1, x2, x3], dim=1)
        out = self.Act(self.ConvCat(out))
        return out * x


class SAFMB(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        self.safm = SAFM(dim)
        self.ccm = CCM(dim, ffn_scale)

    def forward(self, x):
        x = self.safm(self.norm1(x)) + x
        x = self.ccm(self.norm2(x)) + x
        return x



class MFA(nn.Module):
    def __init__(self, dim=33, n_blocks=8, ffn_scale=1.8, upscaling_factor=4):
        super().__init__()
        self.scale = upscaling_factor
        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1)

        self.feats = nn.Sequential(*[SAFMB(dim, ffn_scale) for _ in range(n_blocks)])

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor**2, 3, 1, 1),
            nn.PixelShuffle(upscaling_factor)
        )
    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x) + x
        x = self.to_img(x)
        return x


if __name__== '__main__':
    #############Test Model Complexity #############
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    # x = torch.randn(1, 3, 640, 360)
    # x = torch.randn(1, 3, 427, 240)
    # x = torch.randn(1, 3, 320, 180)
    x = torch.randn(1, 3, 256, 256)

    model = NTIRE_k3d8_sg52(dim=36, n_blocks=8, ffn_scale=1.66, upscaling_factor=4)
    print(model)
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    output = model(x)
    print(output.shape)
