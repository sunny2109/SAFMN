import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
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


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        x = self.gamma * (x * Nx) + self.beta + x
        return x


# Convolutional Channel Mixer
class CCM(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.ccm = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0, bias=False)
        )

    def forward(self, x):
        return self.ccm(x)


# Spatially-Adaptive Feature Modulation
class SAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        # Multiscale feature representation
        self.mfr = nn.ModuleList([nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim, bias=False) for i in range(self.n_levels)])

        # Feature aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)

        # Activation
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]

        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h//2**(i+1), w//2**(i+1))
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.mfr[i](xc[i])
            out.append(s)

        out = self.aggr(torch.cat(out, dim=1))
        # Feature modulation
        out = self.act(out) * x
        return out


class AttBlock(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.safm = SAFM(dim) 
        self.ccm = CCM(dim, ffn_scale)

    def forward(self, x):
        x = self.safm(x) + x
        x = self.ccm(x) + x
        return x


@ARCH_REGISTRY.register()
class SAFMN_NTIRE(nn.Module):
    def __init__(self, dim, n_blocks=8, ffn_scale=2.0, upscaling_factor=4):
        super().__init__()
        self.scale = upscaling_factor
        
        self.norm = GRN(dim)

        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1)

        self.feats = nn.Sequential(*[AttBlock(dim, ffn_scale) for _ in range(n_blocks)])

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor**2, 3, 1, 1),
            nn.PixelShuffle(upscaling_factor)
        )
        
    def forward(self, x):
        ident = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        x = self.to_feat(x)
        x = self.norm(x)
        x = self.feats(x)
        x = self.to_img(x)
        return x + ident


if __name__== '__main__':
    #############Test Model Complexity #############
    # import time
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # x = torch.randn(1, 3, 640, 360)
    # x = torch.randn(1, 3, 427, 240)
    # x = torch.randn(1, 3, 320, 180)#.to(device)
    x = torch.randn(1, 3, 256, 256)

    model = SAFMN_NTIRE(dim=36, n_blocks=8, ffn_scale=2.0, upscaling_factor=4)
    print(model)
    print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    output = model(x)
    print(output.shape)


