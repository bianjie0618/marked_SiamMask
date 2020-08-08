# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.anchors import Anchors


class SiamMask(nn.Module):
    def __init__(self, anchors=None, o_sz=63, g_sz=127):
        super(SiamMask, self).__init__()
        self.anchors = anchors  # anchor_cfg
        self.anchor_num = len(self.anchors["ratios"]) * len(self.anchors["scales"])
        self.anchor = Anchors(anchors)
        self.features = None
        self.rpn_model = None  # self.rpn_model将被子类的self.rpn_model覆盖
        self.mask_model = None  # self.mask_model也将被子类的self.mask_model覆盖
        self.o_sz = o_sz
        self.g_sz = g_sz
        self.upSample = nn.UpsamplingBilinear2d(size=[g_sz, g_sz])

        self.all_anchors = None

    def set_all_anchors(self, image_center, size):
        # cx,cy,w,h
        if not self.anchor.generate_all_anchors(image_center, size):  # 设定所有的anchors，如果设置过，就不用重新设置了
            return
        all_anchors = self.anchor.all_anchors[1]  # cx, cy, w, h
        self.all_anchors = torch.from_numpy(all_anchors).float().cuda()  # 转变成tensor，并推送到GPU上
        self.all_anchors = [self.all_anchors[i] for i in range(4)]

    def feature_extractor(self, x):
        # 参数共享阶段
        return self.features(x)

    def rpn(self, template, search):
        pred_cls, pred_loc = self.rpn_model(template, search)
        return pred_cls, pred_loc

    def mask(self, template, search):
        pred_mask = self.mask_model(template, search)
        return pred_mask

    def _add_rpn_loss(self, label_cls, label_loc, lable_loc_weight, label_mask, label_mask_weight,
                      rpn_pred_cls, rpn_pred_loc, rpn_pred_mask):
        rpn_loss_cls = select_cross_entropy_loss(rpn_pred_cls, label_cls)

        rpn_loss_loc = weight_l1_loss(rpn_pred_loc, label_loc, lable_loc_weight)

        rpn_loss_mask, iou_m, iou_5, iou_7 = select_mask_logistic_loss(rpn_pred_mask, label_mask, label_mask_weight)

        return rpn_loss_cls, rpn_loss_loc, rpn_loss_mask, iou_m, iou_5, iou_7

    def run(self, template, search, softmax=False):
        """
        run network
        """
        template_feature = self.feature_extractor(template)
        search_feature = self.feature_extractor(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(template_feature, search_feature)
        rpn_pred_mask = self.mask(template_feature, search_feature)  # (b, 63*63, w, h)

        if softmax:  # 如果训练，则softmax；在referrence阶段，无需softmax。
            rpn_pred_cls = self.softmax(rpn_pred_cls)
        return rpn_pred_cls, rpn_pred_loc, rpn_pred_mask, template_feature, search_feature

    def softmax(self, cls):
        b, a2, h, w = cls.size()  # 分类是两类。。。这里的a2//2就是anchor的种类，每一个锚点都有a2//2个anchor(一系列)
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()  # 转变成在内存中连续的格式；
        cls = F.log_softmax(cls, dim=4)  # 在dim=4的方向求解softmax并log ; b*k*w*h*2
        return cls

    def forward(self, input):
        """
        :param input: dict of input with keys of:
                'template': [b, 3, h1, w1], input template image.
                'search': [b, 3, h2, w2], input search image.
                'label_cls':[b, max_num_gts, 5] or None(self.training==False),
                                     each gt(ground truth) contains x1,y1,x2,y2,class.
        :return: dict of loss, predict, accuracy
        """
        template = input['template']
        search = input['search']
        if self.training:  # 这块继承的是nn.Module吗？？？
            label_cls = input['label_cls']
            label_loc = input['label_loc']
            lable_loc_weight = input['label_loc_weight']
            label_mask = input['label_mask']
            label_mask_weight = input['label_mask_weight']

        rpn_pred_cls, rpn_pred_loc, rpn_pred_mask, template_feature, search_feature = \
            self.run(template, search, softmax=self.training)

        outputs = dict()

        outputs['predict'] = [rpn_pred_loc, rpn_pred_cls, rpn_pred_mask, template_feature, search_feature]

        if self.training:
            rpn_loss_cls, rpn_loss_loc, rpn_loss_mask, iou_acc_mean, iou_acc_5, iou_acc_7 = \
                self._add_rpn_loss(label_cls, label_loc, lable_loc_weight, label_mask, label_mask_weight,
                                   rpn_pred_cls, rpn_pred_loc, rpn_pred_mask)
            outputs['losses'] = [rpn_loss_cls, rpn_loss_loc, rpn_loss_mask]
            outputs['accuracy'] = [iou_acc_mean, iou_acc_5, iou_acc_7]

        return outputs

    def template(self, z):
        self.zf = self.feature_extractor(z)
        cls_kernel, loc_kernel = self.rpn_model.template(self.zf)   # rpn_model.template是个什么方法？？？
        return cls_kernel, loc_kernel

    def track(self, x, cls_kernel=None, loc_kernel=None, softmax=False):
        xf = self.feature_extractor(x)
        rpn_pred_cls, rpn_pred_loc = self.rpn_model.track(xf, cls_kernel, loc_kernel)
        if softmax:
            rpn_pred_cls = self.softmax(rpn_pred_cls)
        return rpn_pred_cls, rpn_pred_loc


def get_cls_loss(pred, label, select):
    if select.nelement() == 0: return pred.sum()*0.  # 在预测正例/负例时，如果没有正例/负例，直接返回0
    pred = torch.index_select(pred, 0, select)    # 挑选标签为正/负的预测; select是一维的
    label = torch.index_select(label, 0, select)  # 挑选正/负标签值

    return F.nll_loss(pred, label)
    # 就是等同于nn.NLLLoss()负对数似然损失. 为什么没有加log和softmax这一步？？？ 因为前面已经有这一步了，def softmax函数


# rpn_loss_cls
def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)  # 预测种类的数据形状为 k，size，size，2； 标签的形状为：k，size，size
    label = label.view(-1)  # 变成了在第0维方向上和label长度相同的数据，即一维tensor，在这里面存放所有的元素
    # 上面的pred之前为什么把类别放在第4各坐标轴方向，就是为了这里方便view，view只能对连续的张量进行操作。 注意总结view和permute的区别
    # 分别生成正负标签的索引，并通过到GPU上
    pos = Variable(label.data.eq(1).nonzero().squeeze()).cuda()  # nonzero返回的是非零元素的索引，通常返回的是2维tensor
    neg = Variable(label.data.eq(0).nonzero().squeeze()).cuda()  # nonzero返回的每一行表示一个非零元素的索引

    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5

# 用的是l1损失，并非smoothl1损失；计算位置的回归损失
# rpn_loss_loc
def weight_l1_loss(pred_loc, label_loc, loss_weight):
    """
    :param pred_loc: [b, 4k, h, w]
    :param label_loc: [b, 4k, h, w]
    :param loss_weight:  [b, k, h, w]
    :return: loc loss value
    """
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)   # 这种布局值得理解一番
    diff = (pred_loc - label_loc).abs()   # 位置预测其实是位置的变化值预测
    diff = diff.sum(dim=1).view(b, -1, sh, sw)  # 去掉size=1的维度方向，也可以用squeeze()
    loss = diff * loss_weight   # 如何给定loc标签的损失权重？？？
    return loss.sum().div(b)    # 关于维度方向，应该好好理解一番***********************************


# rpn_loss_mask
def select_mask_logistic_loss(p_m, mask, weight, o_sz=63, g_sz=127):
    weight = weight.view(-1)   # weight的size为w*h，每个RoW位置的权重由该位置的最大的类别标签决定
    pos = Variable(weight.data.eq(1).nonzero().squeeze())   # 挑选出正类来
    if pos.nelement() == 0: return p_m.sum() * 0, p_m.sum() * 0, p_m.sum() * 0, p_m.sum() * 0

    # 下面4行是产生m_n的
    p_m = p_m.permute(0, 2, 3, 1).contiguous().view(-1, 1, o_sz, o_sz) # b,w,h,c
    p_m = torch.index_select(p_m, 0, pos)
    p_m = nn.UpsamplingBilinear2d(size=[g_sz, g_sz])(p_m)  # 双线性插值，以减少计算量
    p_m = p_m.view(-1, g_sz * g_sz)   # 之所以这么干，是为了计算损失函数L_mask方便

    # 下面两行产生cn
    # Extracts sliding local blocks from a batched input tensor. 可以当官网查看该函数的帮助文档
    mask_uf = F.unfold(mask, (g_sz, g_sz), padding=32, stride=8)   # 为什么padding要32？？？ 强行解释可以查看table 8，有注释。
    mask_uf = torch.transpose(mask_uf, 1, 2).contiguous().view(-1, g_sz * g_sz)  # 4,127*127,25*25
    # b w c h
    # 紧贴下面一行是产生positive cn的
    mask_uf = torch.index_select(mask_uf, 0, pos)
    loss = F.soft_margin_loss(p_m, mask_uf)        # 论文中的公式(3)，计算L_mask损失函数；只计算正类
    iou_m, iou_5, iou_7 = iou_measure(p_m, mask_uf)
    return loss, iou_m, iou_5, iou_7   # iou_m表示iou，而iou_5, iou_7表示相应的准确率


def iou_measure(pred, label):
    pred = pred.ge(0)  # 返回布尔值；大于等于0的全部为1，小于零的全部为0
    mask_sum = pred.eq(1).add(label.eq(1))    # cool
    intxn = torch.sum(mask_sum == 2, dim=1).float()
    union = torch.sum(mask_sum > 0, dim=1).float()
    iou = intxn/union
    return torch.mean(iou), (torch.sum(iou > 0.5).float()/iou.shape[0]), (torch.sum(iou > 0.7).float()/iou.shape[0])
    

if __name__ == "__main__":
    p_m = torch.randn(4, 63*63, 25, 25)   # 这里的4代表batch；25代表RoW的个数，见论文的figure2，论文中RoW的个数是17
    cls = torch.randn(4, 1, 25, 25) > 0.9  # 真实的数据是怎么作标签的？？？？？
    mask = torch.randn(4, 1, 255, 255) * 2 - 1   # mask_label

    loss = select_mask_logistic_loss(p_m, mask, cls)
    print(loss)
