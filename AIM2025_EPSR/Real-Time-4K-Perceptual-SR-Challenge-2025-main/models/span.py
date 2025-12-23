from collections import OrderedDict
import torch
from torch import nn as nn
import torch.nn.functional as F


class Conv3XC2(nn.Module):
    def __init__(self, c_in, c_out, gain1=1, gain2=0, s=1, groups=1, bias=True, relu=False):
        super(Conv3XC2, self).__init__()
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.stride = s
        self.has_relu = relu
        self.groups = groups
        self.gain = gain1
        gain = gain1

        self.sk = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=1, padding=0, stride=s, bias=bias, groups=groups)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_in * gain, kernel_size=1, padding=0, bias=bias, groups=groups),
            nn.Conv2d(in_channels=c_in * gain, out_channels=c_out * gain, kernel_size=3, stride=s, padding=0, bias=bias, groups=groups),
            nn.Conv2d(in_channels=c_out * gain, out_channels=c_out, kernel_size=1, padding=0, bias=bias, groups=groups),
        )

        self.eval_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1, stride=s, bias=bias, groups=groups)
        self.eval_conv.weight.requires_grad = False
        self.eval_conv.bias.requires_grad = False
        self.update_params()

    def update_params(self):
        # 分组卷积的权重更新
        with torch.no_grad():
            # 获取每个卷积层的权重和偏置
            w1 = self.conv[0].weight.data.clone()
            b1 = self.conv[0].bias.data.clone() if self.conv[0].bias is not None else None
            w2 = self.conv[1].weight.data.clone()
            b2 = self.conv[1].bias.data.clone() if self.conv[1].bias is not None else None
            w3 = self.conv[2].weight.data.clone()
            b3 = self.conv[2].bias.data.clone() if self.conv[2].bias is not None else None
    
            # 对于分组卷积，我们需要对每个组的权重分别进行卷积操作
            # 这里我们假设输入通道和输出通道的数量是组数的整数倍
            group_in_channels = w1.size(1) // self.groups
            group_out_channels = w3.size(0) // self.groups
    
            # 初始化合并后的权重和偏置
            weight_concat = torch.zeros_like(self.eval_conv.weight.data)
            bias_concat = torch.zeros_like(self.eval_conv.bias.data) if self.eval_conv.bias is not None else None
    
            # 对每个组进行操作
            # import pdb; pdb.set_trace()
            for g in range(self.groups):
                # 提取每个组的权重
                w1_g = w1[g * group_out_channels * self.gain:(g + 1) * group_out_channels * self.gain, :, :, :]
                w2_g = w2[g * group_out_channels * self.gain:(g + 1) * group_out_channels * self.gain, 
                          :, :, :]
                # g * group_in_channels * self.gain:(g + 1) * self.gain * group_in_channels
                w3_g = w3[g * group_out_channels:(g + 1) * group_out_channels, :, :, :]
    
                # 计算合并后的权重
                # import pdb; pdb.set_trace()
                w_g = F.conv2d(w1_g.flip(2, 3).permute(1, 0, 2, 3), w2_g, padding=2, stride=1).flip(2, 3).permute(1, 0, 2, 3)
                      
                weight_concat_g = F.conv2d(w_g.flip(2, 3).permute(1, 0, 2, 3), w3_g, padding=0, stride=1).flip(2, 3).permute(1, 0, 2, 3)
    
                # 将每个组的权重放回合并后的权重张量中
                weight_concat[g * group_out_channels:(g + 1) * group_out_channels, :, :, :] = weight_concat_g
    
                # 如果有偏置，也需要进行合并
                if b1 is not None and b2 is not None and b3 is not None:
                    b_g = (w2_g * b1[g * group_out_channels * self.gain:(g + 1) * group_out_channels * self.gain].reshape(1, -1, 1, 1)).sum((1, 2, 3)) + \
                        b2[g * group_out_channels * self.gain:(g + 1) * group_out_channels * self.gain]
                    bias_concat_g = (w3_g * b_g.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3[g * group_out_channels:(g + 1) * group_out_channels]
                    bias_concat[g * group_out_channels:(g + 1) * group_out_channels] = bias_concat_g
    
            # 将shortcut层的权重和偏置加到合并后的权重和偏置上
            sk_w = self.sk.weight.data.clone()
            sk_b = self.sk.bias.data.clone() if self.sk.bias is not None else None
            if sk_w is not None:
                H_pixels_to_pad = (self.eval_conv.kernel_size[0] - 1) // 2
                W_pixels_to_pad = (self.eval_conv.kernel_size[1] - 1) // 2
                sk_w_padded = F.pad(sk_w, [W_pixels_to_pad, W_pixels_to_pad, H_pixels_to_pad, H_pixels_to_pad])
    
                # Add the shortcut weights to the concatenated weights
                # We need to ensure that the shortcut weights are added to the correct group
                # import pdb; pdb.set_trace()
                weight_concat += sk_w_padded
                # for g in range(self.groups):
                #     weight_concat[g * group_in_channels * self.gain:(g + 1) * group_in_channels * self.gain, :, :, :] += \
                #         sk_w_padded[g * group_in_channels * self.gain:(g + 1) * group_in_channels * self.gain, :, :, :]
    
            # If there is a bias in the shortcut, add it to the concatenated bias
            if sk_b is not None:
                bias_concat += sk_b
    
            # Update the eval_conv layer's weights and biases with the concatenated values
            self.eval_conv.weight.data = weight_concat
            if self.eval_conv.bias is not None:
                self.eval_conv.bias.data = bias_concat


    def forward(self, x):
        if self.training:
            pad = 1
            x_pad = F.pad(x, (pad, pad, pad, pad), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            self.update_params()
            out = self.eval_conv(x)

        if self.has_relu:
            out = F.leaky_relu(out, negative_slope=0.05)
        return out

def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value


class ShiftConv2d_4(nn.Module):
    def __init__(self, inp_channels, move_channels=2, move_pixels=1):
        super(ShiftConv2d_4, self).__init__()
        self.inp_channels = inp_channels
        self.move_p = move_pixels
        self.move_c = move_channels
        self.weight = nn.Parameter(torch.zeros(inp_channels, 1, 3, 3), requires_grad=False)
        mid_channel = inp_channels // 2
        up_channels = (mid_channel - move_channels * 2, mid_channel - move_channels)
        down_channels = (mid_channel - move_channels, mid_channel)
        left_channels = (mid_channel, mid_channel + move_channels)
        right_channels = (mid_channel + move_channels, mid_channel + move_channels * 2)
        self.weight[left_channels[0]:left_channels[1], 0, 1, 2] = 1.0  ## left
        self.weight[right_channels[0]:right_channels[1], 0, 1, 0] = 1.0  ## right
        self.weight[up_channels[0]:up_channels[1], 0, 2, 1] = 1.0  ## up
        self.weight[down_channels[0]:down_channels[1], 0, 0, 1] = 1.0  ## down
        self.weight[0:mid_channel - move_channels * 2, 0, 1, 1] = 1.0  ## identity
        self.weight[mid_channel + move_channels * 2:, 0, 1, 1] = 1.0  ## identity

    def forward(self, x):
        for i in range(self.move_p):
            x = F.conv2d(input=x, weight=self.weight, bias=None, stride=1, padding=1, groups=self.inp_channels)

        return x


class BSConvU(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, padding_mode="zeros", with_bn=False, bn_kwargs=None):
        super().__init__()

        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.add_module("pw", torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
        ))

        # batchnorm
        if with_bn:
            self.add_module("bn", torch.nn.BatchNorm2d(num_features=out_channels, **bn_kwargs))

        # depthwise
        self.add_module("dw", torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                padding_mode=padding_mode,
        ))


def conv_layer(in_channels,
               out_channels,
               kernel_size,
               bias=True):
    """
    Re-write convolution layer for adaptive `padding`.
    """
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2),
               int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=padding,
                     bias=bias)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    """
    Activation functions for ['relu', 'lrelu', 'prelu'].
    Parameters
    ----------
    act_type: str
        one of ['relu', 'lrelu', 'prelu'].
    inplace: bool
        whether to use inplace operator.
    neg_slope: float
        slope of negative region for `lrelu` or `prelu`.
    n_prelu: int
        `num_parameters` for `prelu`.
    ----------
    """
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            'activation layer [{:s}] is not found'.format(act_type))
    return layer


def sequential(*args):
    """
    Modules will be added to the a Sequential Container in the order they
    are passed.

    Parameters
    ----------
    args: Definition of Modules in order.
    -------
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError(
                'sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def pixelshuffle_block(in_channels,
                       out_channels,
                       upscale_factor=2,
                       kernel_size=3):
    """
    Upsample features according to `upscale_factor`.
    """
    conv = conv_layer(in_channels,
                      out_channels * (upscale_factor ** 2),
                      kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)

class Conv3XC(nn.Module):
    def __init__(self, c_in, c_out, gain1=1, gain2=0, s=1, bias=True, relu=False):
        super(Conv3XC, self).__init__()
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.stride = s
        self.has_relu = relu
        gain = gain1

        #self.sk = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=1, padding=0, stride=s, bias=bias)
        #self.conv = nn.Sequential(
        #    nn.Conv2d(in_channels=c_in, out_channels=c_in * gain, kernel_size=1, padding=0, bias=bias),
        #    nn.Conv2d(in_channels=c_in * gain, out_channels=c_out * gain, kernel_size=3, stride=s, padding=0, bias=bias),
        #    nn.Conv2d(in_channels=c_out * gain, out_channels=c_out, kernel_size=1, padding=0, bias=bias),
        #)

        self.eval_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1, stride=s, bias=bias)
        self.eval_conv.weight.requires_grad = False
        self.eval_conv.bias.requires_grad = False
        # self.update_params()

    def update_params(self):
        w1 = self.conv[0].weight.data.clone().detach()
        b1 = self.conv[0].bias.data.clone().detach()
        w2 = self.conv[1].weight.data.clone().detach()
        b2 = self.conv[1].bias.data.clone().detach()
        w3 = self.conv[2].weight.data.clone().detach()
        b3 = self.conv[2].bias.data.clone().detach()

        w = F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

        self.weight_concat = F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        sk_w = self.sk.weight.data.clone().detach()
        sk_b = self.sk.bias.data.clone().detach()
        target_kernel_size = 3

        H_pixels_to_pad = (target_kernel_size - 1) // 2
        W_pixels_to_pad = (target_kernel_size - 1) // 2
        sk_w = F.pad(sk_w, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])

        self.weight_concat = self.weight_concat + sk_w
        self.bias_concat = self.bias_concat + sk_b

        self.eval_conv.weight.data = self.weight_concat
        self.eval_conv.bias.data = self.bias_concat


    def forward(self, x):
        out = self.eval_conv(x)

        if self.has_relu:
            out = F.leaky_relu(out, negative_slope=0.05)
        return out
    

class CustomActivation(nn.Module):
    def __init__(self, num_channels):
        super(CustomActivation, self).__init__()
        self.alpha = nn.Parameter(torch.ones((1, num_channels, 1, 1)), requires_grad=True)
        # self.act1 = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        # return x * torch.sigmoid(x)
        # return self.act1(x)
        # return x
        return x * torch.sigmoid(self.alpha * x)


class SlimBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        # dw_channel = c // 2
        dw_channel = c
        self.conv1 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=2, bias=True)
        # self.conv1 = Conv3XC(dw_channel, dw_channel, gain1=2, s=1)
        # self.act = activation('lrelu', neg_slope=0.1, inplace=True)
        self.act = CustomActivation(c)


    def forward(self, inp):

        # x1_init, x2_init = torch.chunk(inp, chunks=2, dim=1)
        x = self.conv1(inp)
        x = self.act(x)
        # x1, x2 = torch.chunk(x, chunks=2, dim=1)
        # y1 = x1_init + x1
        # y2 = x2_init + x2
        #  y = torch.cat([y1, y2], dim=1)
        y = x + inp
        return y

#class SlimBlock(nn.Module):
#    def __init__(self, c, DW_Expand=1, FFN_Expand=1, drop_out_rate=0.):
#        super().__init__()
#        dw_channel = c // 2
#        self.conv1 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
#
#        self.act = nn.ReLU(inplace=True)
#
#
#    def forward(self, inp):
#        x1_init, x2_init = torch.chunk(inp, chunks=2, dim=1)
#        x1 = x1_init
#        x1 = self.conv1(x1)
#        x1 = self.act(x1)
#        y1 = x1_init + x1
#        y = torch.cat([y1, x2_init], dim=1)
#        return y


class SPAB1(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 bias=False):
        super(SPAB1, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.c1_r = Conv3XC(in_channels, mid_channels, gain1=2, s=1)
        self.c2_r = Conv3XC(mid_channels, mid_channels, gain1=2, s=1)
        self.c3_r = Conv3XC(mid_channels, out_channels, gain1=2, s=1)
        self.act1 = torch.nn.SiLU(inplace=True)
        # self.act2 = activation('lrelu', neg_slope=0.1, inplace=True)

    def forward(self, x):
        out1 = (self.c1_r(x))
        out1_act = self.act1(out1)

        out2 = (self.c2_r(out1_act))
        out2_act = self.act1(out2)

        out3 = (self.c3_r(out2_act))

        sim_att = torch.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att

        # return out, out1, sim_att
        return out, out1, sim_att


class SPAB2(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 bias=False):
        super(SPAB2, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.c1_r = Conv3XC2(in_channels, mid_channels, gain1=2, s=1, groups=2)
        self.c2_r = Conv3XC2(mid_channels, mid_channels, gain1=2, s=1, groups=2)
        self.c3_r = Conv3XC2(mid_channels, out_channels, gain1=2, s=1, groups=2)
        self.act1 = CustomActivation(mid_channels)
        self.act2 = CustomActivation(mid_channels)
        self.act = torch.nn.SiLU(inplace=True)
        # self.act2 = activation('lrelu', neg_slope=0.1, inplace=True)

        
        # self.alpha = nn.Parameter(1*torch.ones((1, out_channels, 1, 1)), requires_grad=True)
        # self.beta = nn.Parameter(1*torch.ones((1, out_channels, 1, 1)), requires_grad=True)
        # self.gamma = nn.Parameter(0.5*torch.ones((1, out_channels, 1, 1)), requires_grad=True)

    def forward(self, x):
        out1 = (self.c1_r(x))
        out1_act = self.act1(out1)

        out2 = (self.c2_r(out1_act))
        out2_act = self.act2(out2)

        out3 = (self.c3_r(out2_act))
        out3 = self.act(out3) + x

        # sim_att = torch.sigmoid(self.alpha * out3) - 0.5
        # sim_att = self.alpha * torch.sigmoid(self.beta*out3) - self.gamma
        # out = (out3 + x) * sim_att

        return out3, out1, out3

# @ARCH_REGISTRY.register()
class SPAN30(nn.Module):
    """
    Swift Parameter-free Attention Network for Efficient Super-Resolution
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 feature_channels=48,
                 upscale=4,
                 bias=True,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)
                 ):
        super(SPAN30, self).__init__()

        in_channels = num_in_ch
        out_channels = num_out_ch
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_1 = Conv3XC(in_channels, feature_channels, gain1=2, s=1)
        self.block_1 = SPAB1(feature_channels, bias=bias)
        self.block_2 = SPAB1(feature_channels, bias=bias)
        self.block_3 = SPAB1(feature_channels, bias=bias)
        self.block_4 = SPAB1(feature_channels, bias=bias)
        self.block_5 = SPAB1(feature_channels, bias=bias)
        self.block_6 = SPAB1(feature_channels, bias=bias)

        self.conv_cat = conv_layer(feature_channels * 4, feature_channels, kernel_size=1, bias=True)
        self.conv_2 = Conv3XC(feature_channels, feature_channels, gain1=2, s=1)

        self.upsampler = pixelshuffle_block(feature_channels, out_channels, upscale_factor=upscale)
        self.cuda()(torch.randn(1, 3, 256, 256).cuda())

    def forward(self, x):
        # x = x.type(torch.HalfTensor).cuda()
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        out_feature = self.conv_1(x)

        out_b1, out_b0_2, att1 = self.block_1(out_feature)
        out_b2, out_b1_2, att2 = self.block_2(out_b1)

        out_b3, out_b2_2, att3 = self.block_3(out_b2)
        out_b4, out_b3_2, att4 = self.block_4(out_b3)
        out_b5, out_b4_2, att5 = self.block_5(out_b4)
        out_b6, out_b5_2, att6 = self.block_6(out_b5)

        out_final = self.conv_2(out_b6)
        out = self.conv_cat(torch.cat([out_feature, out_final, out_b1, out_b5_2], 1))
        # out = self.conv_cat(torch.cat([out_feature, out_b4], 1))
        output = self.upsampler(out)

        return output