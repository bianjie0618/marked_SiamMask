from models.siammask import SiamMask
from models.features import MultiStageFeature
from models.rpn import RPN, DepthCorr
from models.mask import Mask
import torch
import torch.nn as nn
from utils.load_helper import load_pretrain
from resnet import resnet50

# 该模块全部搞定，是siammask的核心模块


class ResDownS(nn.Module):
    def __init__(self, inplane, outplane):
        super(ResDownS, self).__init__()
        self.downsample = nn.Sequential(
                nn.Conv2d(inplane, outplane, kernel_size=1, bias=False),
                nn.BatchNorm2d(outplane))

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:  # 为什么这么做？
            l = 4
            r = -4
            x = x[:, :, l:r, l:r]
        return x


class ResDown(MultiStageFeature):
    # 正向下采样阶段(conv1到conv4_x,参考table 8)，此阶段参数共享
    def __init__(self, pretrain=False):
        super(ResDown, self).__init__()
        self.features = resnet50(layer3=True, layer4=False)  # 只取resnet的前3层,这个过程是在搭建框架，参数的具体值并没有给出
        if pretrain:
            load_pretrain(self.features, 'resnet.model')   # 这里载入预训练的参数; 返回来的model传给了谁？？？

        self.downsample = ResDownS(1024, 256)  # 进入了调整层adjust，由1024转变成256

        self.layers = [self.downsample, self.features.layer2, self.features.layer3]   # 这步想干啥？？？
        self.train_nums = [1, 3]
        self.change_point = [0, 0.5]   # 前一半的epoch只训练self.downsample；后一半的epoch训练self.layers里面的全部层

        self.unfix(0.0)

    def param_groups(self, start_lr, feature_mult=1):
        lr = start_lr * feature_mult

        def _params(module, mult=1):
            params = list(filter(lambda x:x.requires_grad, module.parameters()))
            if len(params):
                return [{'params': params, 'lr': lr * mult}]
            else:
                return []

        groups = []
        groups += _params(self.downsample)
        groups += _params(self.features, 0.1)
        return groups

    def forward(self, x):
        output = self.features(x)
        p3 = self.downsample(output[1])
        return p3


class UP(RPN):
    # 此阶段从调整层adjust开始，是头部阶段
    def __init__(self, anchor_num=5, feature_in=256, feature_out=256):
        super(UP, self).__init__()

        self.anchor_num = anchor_num
        self.feature_in = feature_in
        self.feature_out = feature_out

        self.cls_output = 2 * self.anchor_num
        self.loc_output = 4 * self.anchor_num

        # 下面两行本质上是实例化了“两个”分支，分别是rpn的分类分支和回归分支
        self.cls = DepthCorr(feature_in, feature_out, self.cls_output)  # 这里的feature_in, feature_out, self.cls_output是指输入输出的通道数
        self.loc = DepthCorr(feature_in, feature_out, self.loc_output)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        return cls, loc   # 返回分类和回归的结果


class MaskCorr(Mask):
    def __init__(self, oSz=63): # 为什么是63*63？
        super(MaskCorr, self).__init__()
        self.oSz = oSz
        self.mask = DepthCorr(256, 256, self.oSz**2)  # DepthCorr类包含了从调整层adjust到depth-wise-corr再到头部结构hΦ/bσ/sφ

    def forward(self, z, x):
        return self.mask(z, x)  # mask结构也是2个1x1的卷积


class Custom(SiamMask):
    def __init__(self, pretrain=False, **kwargs):
        super(Custom, self).__init__(**kwargs)
        self.features = ResDown(pretrain=pretrain)
        self.rpn_model = UP(anchor_num=self.anchor_num, feature_in=256, feature_out=256)
        self.mask_model = MaskCorr()

    def template(self, template):
        self.zf = self.features(template)

    def track(self, search):
        search = self.features(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(self.zf, search)  # 父类中有定义self.rpn，其实就是self.rpn_model，这里产生了两个分支，分别是cls分支和loc分支
        return rpn_pred_cls, rpn_pred_loc

    def track_mask(self, search):
        search = self.features(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(self.zf, search)
        pred_mask = self.mask(self.zf, search)
        return rpn_pred_cls, rpn_pred_loc, pred_mask

