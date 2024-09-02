import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Gated_Conv_FeedForward(nn.Module):
    def __init__(self, n_feats, mult=1, bias=False, dropout=0.0):
        super().__init__()

        hidden_features = int(n_feats * mult)

        self.project_in = nn.Conv2d(n_feats, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, n_feats, kernel_size=1, bias=bias)

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        x = x * self.scale + shortcut
        return x


class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30, conv=nn.Conv2d, p=0.25):
        super(CAB, self).__init__()

        BSConvS_Kwargs = {}
        if conv.__name__ == 'BSConvS':
            BSConvS_Kwargs = {'p': p}

        self.cab = nn.Sequential(
            conv(num_feat, num_feat // compress_ratio, 3, 1, 1, **BSConvS_Kwargs),
            nn.GELU(),
            conv(num_feat // compress_ratio, num_feat, 3, 1, 1, **BSConvS_Kwargs),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class GroupGLKA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        i_feats = 2 * n_feats

        self.n_feats = n_feats
        self.i_feats = i_feats

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        self.CAB = CAB(num_feat=n_feats, compress_ratio=3, squeeze_factor=30, conv=BSConvU)

        # Multiscale Large Kernel Attention
        self.LKA7 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 7, 1, 7 // 2, groups=n_feats // 3),  # DW卷积
            nn.Conv2d(n_feats // 3, n_feats // 3, 9, stride=1, padding=(9 // 2) * 4, groups=n_feats // 3, dilation=4),
            # DWD卷积
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))
        self.LKA5 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 5, 1, 5 // 2, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 7, stride=1, padding=(7 // 2) * 3, groups=n_feats // 3, dilation=3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))
        self.LKA3 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 5, stride=1, padding=(5 // 2) * 2, groups=n_feats // 3, dilation=2),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))

        self.LKA = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 9, 1, 9 // 2, groups=n_feats),
            nn.Conv2d(n_feats, n_feats, 11, stride=1, padding=(11 // 2) * 5, groups=n_feats, dilation=5),
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))

        self.X15 = nn.Conv2d(n_feats, n_feats, 3, 1, 1, groups=n_feats)

        self.X3 = nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1, groups=n_feats // 3)
        self.X5 = nn.Conv2d(n_feats // 3, n_feats // 3, 5, 1, 5 // 2, groups=n_feats // 3)
        self.X7 = nn.Conv2d(n_feats // 3, n_feats // 3, 7, 1, 7 // 2, groups=n_feats // 3)

        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, i_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))

    def forward(self, x):
        shortcut = x.clone()

        x = self.norm(x)

        # 在这里引入一个通道注意力分支，可以做成常规的通道注意力分支，也可以做成高效的通道注意力分支
        x_cab = x.clone()
        x_cab = self.CAB(x_cab)

        x = self.proj_first(x)  # 为文中的PW卷积   将通道数增加一倍

        a, x = torch.chunk(x, 2, dim=1)

        a_1, a_2, a_3 = torch.chunk(a, 3, dim=1)
        #a = self.LKA(a) * self.X15(a)
        a = torch.cat([self.LKA3(a_1) * self.X3(a_1), self.LKA5(a_2) * self.X5(a_2), self.LKA7(a_3) * self.X7(a_3)],
                      dim=1)
        x = self.proj_last(x * a) * self.scale + shortcut + x_cab * 0.01  # 这个参数是可以调节的，但是这里的调节不是指可以参加反向传播运算

        return x


class MAB(nn.Module):
    def __init__(
            self, n_feats):
        super().__init__()

        self.LKA = GroupGLKA(n_feats)

        self.LFE = Gated_Conv_FeedForward(n_feats)

    def forward(self, x, pre_attn=None, RAA=None):
        # large kernel attention
        x = self.LKA(x)

        # local feature extraction
        x = self.LFE(x)

        return x


class ResGroup(nn.Module):
    def __init__(self, n_resblocks, n_feats, res_scale=1.0):
        super(ResGroup, self).__init__()
        self.body = nn.ModuleList([
            MAB(n_feats) \
            for _ in range(n_resblocks)])

        self.body_t = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

        # 添加两个高效的注意力模块
        # self.esa = ESA(n_feats, conv=BSConvU)

        # self.cca = CCALayer(n_feats)

    def forward(self, x):
        res = x.clone()

        for i, block in enumerate(self.body):
            res = block(res)

            # if (i + 1) % 6 == 0:
            # res = self.cca(res)
            # res = self.esa(res)

        # x = self.body_t(res) + x
        x = self.body_t(res) + x
        return x


class SoftBSConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=True,
                 ):
        super().__init__()
        # 1×1Conv
        self.sp1 = nn.Sequential(nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            groups=1,
            bias=False,
        ),
            nn.SiLU(), )
        self.act = nn.SiLU()
        # 3×3Conv
        self.sp3 = nn.Sequential(nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=1,
            bias=bias,
        ),
            nn.SiLU(), )
        self.act = nn.SiLU()

    def forward(self, spb):
        spb = self.sp1(spb) + spb
        spb = self.sp3(spb) + spb
        spb = self.act(spb)
        return spb


# =========================================添加两个高效的注意力模块=======================================================
class ESA(nn.Module):
    def __init__(self, num_feat=50, conv=nn.Conv2d, p=0.25):
        super(ESA, self).__init__()
        f = num_feat // 4
        BSConvS_kwargs = {}
        if conv.__name__ == 'BSConvS':
            BSConvS_kwargs = {'p': p}
        self.conv1 = nn.Conv2d(num_feat, f, 1)
        self.conv_f_2 = nn.Conv2d(f, f, 1)
        self.conv_f = SoftBSConv(f, f)
        self.maxPooling = nn.MaxPool2d(kernel_size=7, stride=3)
        self.conv_max = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv2 = conv(f, f, 3, 2, 0)
        self.conv3 = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv3_ = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv4 = nn.Conv2d(f, num_feat, 1)
        self.sigmoid = nn.Sigmoid()
        self.GELU = nn.GELU()

    def forward(self, input):
        c1_ = (self.conv1(input))  # conv1减小特征的通道维度
        c1 = self.conv2(c1_)  # conv2减小特征的空间维度，这里面的所有卷积都被替换成蓝图可分离卷积，算是一个创新点
        v_max = self.maxPooling(c1)  # 继续减小空间维度
        v_range = self.GELU(self.conv_max(v_max))  # debug时进入的还是BSConvU类，但是我看这个属性实现时有**BSConvS_kwargs
        c3 = self.GELU(self.conv3(v_range))
        c3 = self.conv3_(c3)  # 上面的三行代码对应原文中的Conv Groups，用于提取特征，它们都不改变特征的空间维度
        c3 = F.interpolate(c3, (input.size(2), input.size(3)), mode='bilinear', align_corners=False)  # 还原特征的空间维度
        cf = self.conv_f(c1_)
        cf2 = self.conv_f_2(c1_)
        c4 = self.conv4((c3 + cf + cf2))
        m = self.sigmoid(c4)

        return input * m


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class BSConvU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.pw = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # depthwise
        self.dw = torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)  # 作者所用的逐点卷积和深度卷积都不会改变空间维度，但是会改变通道维度
        fea = self.dw(fea)
        return fea


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


@ARCH_REGISTRY.register()
class LDMN(nn.Module):
    def __init__(self, n_resblocks=24, n_resgroups=1, n_colors=3, n_feats=60, scale=4, res_scale=1.0):
        super(LDMN, self).__init__()

        # res_scale = res_scale
        self.n_resgroups = n_resgroups

        self.sub_mean = MeanShift(1.0)
        self.head = nn.Conv2d(n_colors, n_feats, 3, 1, 1)

        # define body module
        self.body = nn.ModuleList([
            ResGroup(
                n_resblocks, n_feats, res_scale=res_scale)
            for i in range(n_resgroups)])

        if self.n_resgroups > 1:
            self.body_t = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

        # define tail module
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_colors * (scale ** 2), 3, 1, 1),
            nn.PixelShuffle(scale)
        )
        self.add_mean = MeanShift(1.0, sign=1)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = x
        for i in self.body:
            res = i(res)

        if self.n_resgroups > 1:
            res = self.body_t(res) + x
        x = self.tail(res)
        x = self.add_mean(x)
        return x

    def visual_feature(self, x):
        fea = []
        x = self.head(x)
        res = x

        for i in self.body:
            temp = res
            res = i(res)
            fea.append(res)

        res = self.body_t(res) + x

        x = self.tail(res)
        return x, fea

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = torch.rand(2, 3, 64, 64).to(device)
    net = LDMN().to(device)

    output = net(input)
    print(output.size())
    model = LDMN()
    print(model)


if __name__ == '__main__':
    main()