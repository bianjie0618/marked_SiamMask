# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F


class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

    def template(self, template):
        raise NotImplementedError

    def track(self, search):
        raise NotImplementedError

    def param_groups(self, start_lr, feature_mult=1, key=None):
        if key is None:
            params = filter(lambda x:x.requires_grad, self.parameters())
        else:
            params = [v for k, v in self.named_parameters() if (key in k) and v.requires_grad]
            # named_parameters()返回的是iterator，iterator返回的是tuple，tuple里面包含name of the parameters和相应的value
        params = [{'params': params, 'lr': start_lr * feature_mult}]
        return params

# depth-wise cross correlation，逐层互相关，即逐层卷积
def conv2d_dw_group(x, kernel):
    batch, channel = kernel.shape[:2]
    x = x.view(1, batch*channel, x.size(2), x.size(3))  # 1 * (b*c) * k * k
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))  # (b*c) * 1 * H * W 这种变形妙
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out


class DepthCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(DepthCorr, self).__init__()
        # adjust layer for asymmetrical features 调整层
        # 把exemplar patch经过调整层的内容记作是conv_kernel
        # 把search patch经过调整层的内容记作是conv_search
        # 在经过adjust后，又经过一个3x3的卷积才进入逐层互相关DepthCorr&&&&&&&&&&&&&&&&&注意本行说明&&&&&&&&&&&&&&&&&&&&&&
        self.conv_kernel = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.conv_search = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )

        # self.head表达的是PRN的分类或回归分支，是由2个1x1的卷积组成
        self.head = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),  # 第一个1x1卷积
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels, kernel_size=1)   # 第二个1x1卷积
                )

    def forward_corr(self, kernel, input):
        kernel = self.conv_kernel(kernel)
        input = self.conv_search(input)
        feature = conv2d_dw_group(input, kernel)
        return feature

    def forward(self, kernel, search):
        feature = self.forward_corr(kernel, search)
        out = self.head(feature)
        return out
