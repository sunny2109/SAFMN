import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SimpleSAFM(nn.Module):
    def __init__(self, dim):
        super().__init__()
  
        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        self.dwconv = nn.Conv2d(dim//2, dim//2, 3, 1, 1, groups=dim//2, bias=False)
        self.out = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]

        x0, x1 = self.proj(x).chunk(2, dim=1)

        x2 = F.adaptive_max_pool2d(x0, (h//8, w//8))
        x2 = self.dwconv(x2)
        x2 = F.interpolate(x2, size=(h, w), mode='bilinear')
        x2 = self.act(x2) * x0

        x = torch.cat([x1, x2], dim=1)
        x = self.out(self.act(x))
        return x


class CCM(nn.Module):
    def __init__(self, dim, ffn_scale):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(dim, int(dim*ffn_scale), 3, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(int(dim*ffn_scale), dim, 1, 1, 0, bias=False)
        )

    def forward(self, x):
        return self.conv(x)



class AttBlock(nn.Module):
    def __init__(self, dim, ffn_scale):
        super().__init__()

        self.conv1 = SimpleSAFM(dim)
        self.conv2 = CCM(dim, ffn_scale)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out


class SAFMN_VIS24(nn.Module):
    def __init__(self, dim=8, n_blocks=1, ffn_scale=2.0, upscaling_factor=4):
        super().__init__()
        self.scale = upscaling_factor

        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1, bias=False)

        self.feats = nn.Sequential(*[AttBlock(dim, ffn_scale) for _ in range(n_blocks)])

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor**2, 3, 1, 1, bias=False),
            nn.PixelShuffle(upscaling_factor)
        )
        
    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x) + x
        return self.to_img(x)



if __name__== '__main__':
    #############Test Model Complexity #############
    # import time
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # x = torch.randn(1, 3, 640, 360)
    # x = torch.randn(1, 3, 427, 240)
    # x = torch.randn(1, 3, 320, 180)#.to(device)
    x = torch.randn(1, 3, 256, 256)

    model = SAFMN_VIS24(dim=8, n_blocks=1, ffn_scale=2.0, upscaling_factor=4)
    print(model)
    print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    output = model(x)
    print(output.shape)

    # x = torch.randn(3, 18, 23, 19)
    # model = IEFS(3, 0.8)
    # print(f'input: {x.shape}, out: {model(x).shape}')


