# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from __future__ import division
from torch.utils.data import Dataset
import numpy as np
import json
import random
import logging
from os.path import join
from utils.bbox_helper import *
from utils.anchors import Anchors
import math
import sys
pyv = sys.version[0]  # 用来查看Python版本
import cv2
if pyv[0] == '3':  #
    cv2.ocl.setUseOpenCL(False)   # OpenCL是Open Computing Language，开放运算语言；为了避免与CUDA冲突

logger = logging.getLogger('global')


sample_random = random.Random()
sample_random.seed(123456)


class SubDataSet(object):
    def __init__(self, cfg):
        for string in ['root', 'anno']:
            if string not in cfg:
                raise Exception('SubDataSet need "{}"'.format(string))

        with open(cfg['anno']) as fin:
            logger.info("loading " + cfg['anno'])  # 记录器记录载入标签数据的信息
            self.labels = self.filter_zero(json.load(fin), cfg)

            def isint(x):
                try:
                    int(x)
                    return True
                except:
                    return False

            # add frames args into labels
            to_del = []
            for video in self.labels:
                for track in self.labels[video]:   # track代表追踪的对象
                    frames = self.labels[video][track]
                    frames = list(map(int, filter(lambda x: isint(x), frames.keys())))   # 将帧名转变成整数
                    frames.sort()
                    self.labels[video][track]['frames'] = frames    # 在特定追踪对象上增加了一副键值对
                    if len(frames) <= 0:
                        logger.info("warning {}/{} has no frames.".format(video, track))
                        to_del.append((video, track))

            # delete tracks with no frames
            for video, track in to_del:
                del self.labels[video][track]  # track表示该视频的追踪号，即第几号追踪对象

            # delete videos with no valid track
            to_del = []
            for video in self.labels:
                if len(self.labels[video]) <= 0:
                    logger.info("warning {} has no tracks".format(video))
                    to_del.append(video)

            for video in to_del:
                del self.labels[video]   # 只有3000视频序列有标注

            self.videos = list(self.labels.keys())

            logger.info(cfg['anno'] + " loaded.")

        # default args
        self.root = "/"
        self.start = 0
        self.num = len(self.labels)   # 视频序列的个数
        self.num_use = self.num
        self.frame_range = 100
        self.mark = "vid"
        self.path_format = "{}.{}.{}.jpg"
        self.mask_format = "{}.{}.m.png"

        self.pick = []

        # input args
        self.__dict__.update(cfg)

        self.has_mask = self.mark in ['coco', 'ytb_vos']    # 只有coco和ytb_vos里有mask； self.has_mask返回的是布尔值

        self.num_use = int(self.num_use)

        # shuffle
        self.shuffle()

    def filter_zero(self, anno, cfg):
        # 过滤掉伪Bbox，即w == 0 or h == 0的bobx
        name = cfg.get('mark', '')  # 从cfg中获取key='mark'的value，否则返回空字符

        out = {}
        tot = 0   # 记录有多少bbox
        new = 0   # 记录有多少真Bbox
        zero = 0  # 记录有多少伪Bbox

        for video, tracks in anno.items():   # .items是获取键值对的方法；获取某个video
            new_tracks = {}
            for trk, frames in tracks.items():   # trk代表当是第几号追踪对象，frames代表的是该追踪对象下的全部帧(属于某一video的)
                new_frames = {}
                for frm, bbox in frames.items():  # 获取特定帧(frm代表的是帧名)里面的若干个annotation
                    tot += 1
                    if len(bbox) == 4:   # 每一帧只有一个Bbox？ 不是的，每一帧有若干个要跟踪的对象(如果个annotation)
                        x1, y1, x2, y2 = bbox
                        w, h = x2 - x1, y2 - y1
                    else:
                        w, h = bbox
                    if w == 0 or h == 0:
                        logger.info('Error, {name} {video} {trk} {bbox}'.format(**locals()))
                        zero += 1
                        continue
                    new += 1   # 如果能跑到这一步，就说明Bbox的长宽都不为0
                    new_frames[frm] = bbox

                if len(new_frames) > 0:
                    new_tracks[trk] = new_frames

            if len(new_tracks) > 0:
                out[video] = new_tracks

        return out

    def log(self):
        logger.info('SubDataSet {name} start-index {start} select [{select}/{num}] path {format}'.format(
            name=self.mark, start=self.start, select=self.num_use, num=self.num, format=self.path_format
        ))

    def shuffle(self):
        lists = list(range(self.start, self.start + self.num))

        m = 0
        pick = []
        while m < self.num_use:
            sample_random.shuffle(lists)   # 就地洗牌
            pick += lists
            m += self.num

        self.pick = pick[:self.num_use]
        return self.pick    # 如何抽取数据

    def get_image_anno(self, video, track, frame):
        frame = "{:06d}".format(frame)   # 传进来的frame是整数，在这里转变成了字符转
        image_path = join(self.root, video, self.path_format.format(frame, track, 'x'))
        image_anno = self.labels[video][track][frame]    # 如果索引coco数据集，那索引的就是train2017.json

        mask_path = join(self.root, video, self.mask_format.format(frame, track))

        return image_path, image_anno, mask_path    # image_anno返回的就是相应的Bbox的坐标

    def get_positive_pair(self, index):
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = random.choice(list(video.keys()))
        track_info = video[track]

        frames = track_info['frames']

        if 'hard' not in track_info:
            template_frame = random.randint(0, len(frames)-1)

            left = max(template_frame - self.frame_range, 0)
            right = min(template_frame + self.frame_range, len(frames)-1) + 1
            search_range = frames[left:right]
            template_frame = frames[template_frame]
            search_frame = random.choice(search_range)
        else:
            search_frame = random.choice(track_info['hard'])   # 硬例挖掘
            left = max(search_frame - self.frame_range, 0)
            right = min(search_frame + self.frame_range, len(frames)-1) + 1  # python [left:right+1) = [left:right]
            template_range = frames[left:right]
            template_frame = random.choice(template_range)
            search_frame = frames[search_frame]

        return self.get_image_anno(video_name, track, template_frame), \
               self.get_image_anno(video_name, track, search_frame)

    def get_random_target(self, index=-1):
        # 此函数的作用是：找到指定索引的video里面某一个(没有特指)追踪对象，然后从这个追踪对象的所有帧里面随机选取一帧，最终获取这一阵的标注信息
        if index == -1:
            index = random.randint(0, self.num-1)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = random.choice(list(video.keys()))
        track_info = video[track]

        frames = track_info['frames']   # 这里的'frames' key是SubDataSet初始化时候增添的。初始化时候还增添了videos属性，方便索引
        frame = random.choice(frames)

        return self.get_image_anno(video_name, track, frame)


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    bbox = [float(x) for x in bbox]
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


class Augmentation:
    def __init__(self, cfg):
        # default args
        self.shift = 0
        self.scale = 0
        self.blur = 0  # False
        self.resize = False
        self.rgbVar = np.array([[-0.55919361,  0.98062831, - 0.41940627],
            [1.72091413,  0.19879334, - 1.82968581],
            [4.64467907,  4.73710203, 4.88324118]], dtype=np.float32)   # 随机给的？
        self.flip = 0

        self.eig_vec = np.array([
            [0.4009, 0.7192, -0.5675],
            [-0.8140, -0.0045, -0.5808],
            [0.4203, -0.6948, -0.5836],
        ], dtype=np.float32)                                              # 特征向量
        self.eig_val = np.array([[0.2175, 0.0188, 0.0045]], np.float32)   # 特征值

        self.__dict__.update(cfg)

    @staticmethod
    def random():   # 静态方法
        # 该方法的作用是生成[-1,1)之间的随机数
        return random.random() * 2 - 1.0   # random.random()返回随机生成的实数，在[0,1)范围内

    def blur_image(self, image):
        def rand_kernel():
            size = np.random.randn(1)
            size = int(np.round(size)) * 2 + 1
            if size < 0: return None
            if random.random() < 0.5: return None
            size = min(size, 45)
            kernel = np.zeros((size, size))
            c = int(size/2)
            wx = random.random()
            kernel[:, c] += 1. / size * wx
            kernel[c, :] += 1. / size * (1-wx)
            return kernel

        kernel = rand_kernel()

        if kernel is not None:
            image = cv2.filter2D(image, -1, kernel)
        return image

    def __call__(self, image, bbox, size, gray=False, mask=None):
        if gray:
            grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.zeros((grayed.shape[0], grayed.shape[1], 3), np.uint8)
            image[:, :, 0] = image[:, :, 1] = image[:, :, 2] = grayed

        shape = image.shape
        # crop_bbox指的是exemplar_size patch，是bbox的母图，这里crop_bbox=127
        crop_bbox = center2corner((shape[0]//2, shape[1]//2, size-1, size-1))  # 为什么size要-1？这是因为长度是按照pixel个数计算

        param = {}
        if self.shift:
            param['shift'] = (Augmentation.random() * self.shift, Augmentation.random() * self.shift)

        if self.scale:
            param['scale'] = ((1.0 + Augmentation.random() * self.scale), (1.0 + Augmentation.random() * self.scale))

        crop_bbox, _ = aug_apply(Corner(*crop_bbox), param, shape)  # crop_bbox经过 移位 和 尺度缩放 来扩增丰富性

        x1 = crop_bbox.x1
        y1 = crop_bbox.y1  # 这两行就如同给出参考坐标的原点

        bbox = BBox(bbox.x1 - x1, bbox.y1 - y1,
                    bbox.x2 - x1, bbox.y2 - y1)    # 这一步确定bbox的相对坐标

        if self.scale:
            scale_x, scale_y = param['scale']
            bbox = Corner(bbox.x1 / scale_x, bbox.y1 / scale_y, bbox.x2 / scale_x, bbox.y2 / scale_y)

        image = crop_hwc(image, crop_bbox, size)   # 这一步就裁剪出具体的exemplar_size或search_size大小的图片
        if not mask is None:
            mask = crop_hwc(mask, crop_bbox, size)

        offset = np.dot(self.rgbVar, np.random.randn(3, 1))
        offset = offset[::-1]  # bgr 2 rgb
        offset = offset.reshape(3)
        image = image - offset

        if self.blur > random.random():
            image = self.blur_image(image)

        if self.resize:
            imageSize = image.shape[:2]
            ratio = max(math.pow(random.random(), 0.5), 0.2)  # 25 ~ 255
            rand_size = (int(round(ratio*imageSize[0])), int(round(ratio*imageSize[1])))
            image = cv2.resize(image, rand_size)
            image = cv2.resize(image, tuple(imageSize))

        if self.flip and self.flip > Augmentation.random():  # 即便self.flip=True，翻不翻转都是随机的
            image = cv2.flip(image, 1)  # 1表示水平翻转
            mask = cv2.flip(mask, 1)
            width = image.shape[1]
            bbox = Corner(width - 1 - bbox.x2, bbox.y1, width - 1 - bbox.x1, bbox.y2)

        return image, bbox, mask


class AnchorTargetLayer:
    def __init__(self, cfg):
        self.thr_high = 0.6  # 应该指的是iou的threshold
        self.thr_low = 0.3
        self.negative = 16
        self.rpn_batch = 64
        self.positive = 16

        self.__dict__.update(cfg)  # 更新里面的属性值(包括属性)，这个__dict__.update很牛逼

    def __call__(self, anchor, target, size, neg=False, need_iou=False):  # 有了__call__,就可以像调用函数一样调用该类的实例对象
        anchor_num = anchor.anchors.shape[0]  # anchors的第零维长度代表anchors的种类个数

        cls = np.zeros((anchor_num, size, size), dtype=np.int64)  # 可以参考anchors.py文件，anchor_num指的是anchor的种类个数
        cls[...] = -1  # -1 ignore 0 negative 1 positive  ...代表取所有的值
        delta = np.zeros((4, anchor_num, size, size), dtype=np.float32)  # 这里的4代表cx,cy,h,w; delta代表他们的改变量
        delta_weight = np.zeros((anchor_num, size, size), dtype=np.float32)  # 位置权重

        def select(position, keep_num=16): # 随机选择最多16个negative
            num = position[0].shape[0]
            if num <= keep_num:
                return position, num
            slt = np.arange(num)
            np.random.shuffle(slt)  # 就地洗牌in-place
            slt = slt[:keep_num]
            return tuple(p[slt] for p in position), keep_num

        if neg:  # 什么时候会出现neg为True的情况？？？就是我们采样的是一对负实例
            l = size // 2 - 3  # left，right
            r = size // 2 + 3 + 1

            cls[:, l:r, l:r] = 0
            # np.where返回满足条件的元素的位置坐标;从高维到低维的顺序返回坐标值，可以参看自己收藏的np.where
            neg, neg_num = select(np.where(cls == 0), self.negative)
            cls[:] = -1
            cls[neg] = 0

            if not need_iou:
                return cls, delta, delta_weight
            else:
                overlap = np.zeros((anchor_num, size, size), dtype=np.float32)
                return cls, delta, delta_weight, overlap

        tcx, tcy, tw, th = corner2center(target)

        anchor_box = anchor.all_anchors[0]   # 查看anchors.py就可以发现，返回来的是两种anchors的表示方法
        anchor_center = anchor.all_anchors[1]
        x1, y1, x2, y2 = anchor_box[0], anchor_box[1], anchor_box[2], anchor_box[3]
        cx, cy, w, h = anchor_center[0], anchor_center[1], anchor_center[2], anchor_center[3]

        # delta  rpn的box分支预测的并不是实际的绝对坐标，而是相对偏移量/变化量。所以这里的delta就是loc_label
        delta[0] = (tcx - cx) / w  # tcx表达了1个GT-box。但这里面算的是所有anchors的delta值
        delta[1] = (tcy - cy) / h
        delta[2] = np.log(tw / w)  # 为什么要用log？？？
        delta[3] = np.log(th / h)

        # IoU
        overlap = IoU([x1, y1, x2, y2], target)

        pos = np.where(overlap > self.thr_high)  # 正例是overlap大于thr_high的位置
        neg = np.where(overlap < self.thr_low)  # 负例是overlap小于thr_low的位置;正例和负例应该互为补集

        pos, pos_num = select(pos, self.positive)  # 正例最多16个
        neg, neg_num = select(neg, self.rpn_batch - pos_num) # 负例个数的选择策略
        # 从这里也可以看出，正负标签是一个相对的概念，主要是通过IoU threshold来判定的
        cls[pos] = 1
        delta_weight[pos] = 1. / (pos_num + 1e-6)  # 正例的权重做平均;其实也是位置权重

        cls[neg] = 0

        if not need_iou:
            return cls, delta, delta_weight   # 返回的是cls_label、loc_label、loc_weight
        else:
            return cls, delta, delta_weight, overlap


class DataSets(Dataset):
    def __init__(self, cfg, anchor_cfg, num_epoch=1):
        super(DataSets, self).__init__()  # 继承自torch.utils.data.Dataset
        global logger  # 全局记录器
        logger = logging.getLogger('global') # 实例化一个叫做‘global’的记录器

        # anchors
        self.anchors = Anchors(anchor_cfg)  # 实例化Anchors

        # size
        self.template_size = 127
        self.origin_size = 127
        self.search_size = 255
        self.size = 17          # 这说明anchor要在depth-wise方式的xcorr之后规划
        self.base_size = 0    # base_size不知道
        self.crop_size = 0

        if 'template_size' in cfg:
            self.template_size = cfg['template_size']
        if 'origin_size' in cfg:
            self.origin_size = cfg['origin_size']
        if 'search_size' in cfg:
            self.search_size = cfg['search_size']
        if 'base_size' in cfg:
            self.base_size = cfg['base_size']
        if 'size' in cfg:
            self.size = cfg['size']
        # 大概理解了下面的一行，可以结合Table8 进行分析，其实是把对examplar的操作迁移到了search image上，只不过要把没有被覆盖的search image区域继续做前面的stride = 8的操作
        if (self.search_size - self.template_size) / self.anchors.stride + 1 + self.base_size != self.size:
            raise Exception("size not match!")  # TODO: calculate size online
        if 'crop_size' in cfg:
            self.crop_size = cfg['crop_size']
        self.template_small = False
        if 'template_small' in cfg and cfg['template_small']:
            self.template_small = True

        self.anchors.generate_all_anchors(im_c=self.search_size//2, size=self.size) # 调用了Anchors类里面的一个方法获取全部的anchors，这一步很重要

        if 'anchor_target' not in cfg:
            cfg['anchor_target'] = {}
        self.anchor_target = AnchorTargetLayer(cfg['anchor_target'])  # 实例化AnchorTargetLayer

        # data sets
        if 'datasets' not in cfg:
            raise(Exception('DataSet need "{}"'.format('datasets')))

        self.all_data = []
        start = 0
        self.num = 0
        for name in cfg['datasets']:
            dataset = cfg['datasets'][name]
            dataset['mark'] = name
            dataset['start'] = start

            dataset = SubDataSet(dataset)  # subdataset是用来干啥的？？？
            dataset.log()
            self.all_data.append(dataset)

            start += dataset.num  # real video number
            self.num += dataset.num_use  # the number used for subset shuffle

        # data augmentation
        aug_cfg = cfg['augmentation']   # 具体看数据扩增是怎么做的？还没弄明白
        self.template_aug = Augmentation(aug_cfg['template'])
        self.search_aug = Augmentation(aug_cfg['search'])
        self.gray = aug_cfg['gray']
        self.neg = aug_cfg['neg']
        self.inner_neg = 0 if 'inner_neg' not in aug_cfg else aug_cfg['inner_neg']

        self.pick = None  # list to save id for each img
        if 'num' in cfg:  # number used in training for all dataset
            self.num = int(cfg['num'])
        self.num *= num_epoch
        self.shuffle()

        self.infos = {
                'template': self.template_size,
                'search': self.search_size,
                'template_small': self.template_small,
                'gray': self.gray,
                'neg': self.neg,
                'inner_neg': self.inner_neg,
                'crop_size': self.crop_size,
                'anchor_target': self.anchor_target.__dict__,
                'num': self.num // num_epoch
                }
        logger.info('dataset informations: \n{}'.format(json.dumps(self.infos, indent=4)))

    def imread(self, path):
        img = cv2.imread(path)
        # origin_size指通过标签Bbox算出来的exemplar patch的实际大小，但需resize为template_size=127
        if self.origin_size == self.template_size:
            return img, 1.0

        def map_size(exe, size):
            return int(round(((exe + 1) / (self.origin_size + 1) * (size+1) - 1)))

        nsize = map_size(self.template_size, img.shape[1])

        img = cv2.resize(img, (nsize, nsize))

        return img, nsize / img.shape[1]

    def shuffle(self):
        pick = []
        m = 0
        while m < self.num:
            p = []
            for subset in self.all_data:
                sub_p = subset.shuffle()
                p += sub_p

            sample_random.shuffle(p)

            pick += p
            m = len(pick)
        self.pick = pick
        logger.info("shuffle done!")
        logger.info("dataset length {}".format(self.num))

    def __len__(self):
        return self.num

    def find_dataset(self, index):
        for dataset in self.all_data:
            if dataset.start + dataset.num > index:
                return dataset, index - dataset.start

    def __getitem__(self, index, debug=False):
        index = self.pick[index]
        dataset, index = self.find_dataset(index)

        gray = self.gray and self.gray > random.random()
        neg = self.neg and self.neg > random.random()

        if neg:
            template = dataset.get_random_target(index)
            if self.inner_neg and self.inner_neg > random.random():
                search = dataset.get_random_target()    # inner_neg == True，从coco或ytb_vos内部选一对 negative pair
            else:
                search = random.choice(self.all_data).get_random_target()
        else:
            template, search = dataset.get_positive_pair(index)   # template = (image_path, image_anno, mask_path)

        def center_crop(img, size):
            shape = img.shape[1]
            if shape == size: return img
            c = shape // 2
            l = c - size // 2
            r = c + size // 2 + 1
            return img[l:r, l:r]

        template_image, scale_z = self.imread(template[0])   # template_image的shape是 511,511,3

        if self.template_small:   # 似乎感觉template_small和origin_size有点关系；
            template_image = center_crop(template_image, self.template_size)

        search_image, scale_x = self.imread(search[0])

        if dataset.has_mask and not neg:
            search_mask = (cv2.imread(search[2], 0) > 0).astype(np.float32)
        else:
            search_mask = np.zeros(search_image.shape[:2], dtype=np.float32)

        if self.crop_size > 0:    # crop_size指什么？？？
            search_image = center_crop(search_image, self.crop_size)
            search_mask = center_crop(search_mask, self.crop_size)

        def toBBox(image, shape):
            # 转变成相对于self.template_size=127条件下的bbox
            imh, imw = image.shape[:2]
            if len(shape) == 4:
                w, h = shape[2]-shape[0], shape[3]-shape[1]
            else:
                w, h = shape
            context_amount = 0.5
            exemplar_size = self.template_size  # 127
            wc_z = w + context_amount * (w+h)
            hc_z = h + context_amount * (w+h)
            s_z = np.sqrt(wc_z * hc_z)
            scale_z = exemplar_size / s_z
            w = w*scale_z
            h = h*scale_z

            # 为什么bbox将中心取在图片的中心？ 因为在预处理阶段已经将bbox的中心置于本函数的arg：image的中心
            # 可以查看预处理阶段的par_crop.py文件
            cx, cy = imw//2, imh//2
            bbox = center2corner(Center(cx, cy, w, h))
            return bbox

        template_box = toBBox(template_image, template[1])
        search_box = toBBox(search_image, search[1])

        template, _, _ = self.template_aug(template_image, template_box, self.template_size, gray=gray)
        search, bbox, mask = self.search_aug(search_image, search_box, self.search_size, gray=gray, mask=search_mask)
        # search分支需要bbox和mask作为label，即标准答案

        def draw(image, box, name):
            image = image.copy()
            x1, y1, x2, y2 = map(lambda x: int(round(x)), box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0))
            cv2.imwrite(name, image)

        if debug:
            draw(template_image, template_box, "debug/{:06d}_ot.jpg".format(index))
            draw(search_image, search_box, "debug/{:06d}_os.jpg".format(index))
            draw(template, _, "debug/{:06d}_t.jpg".format(index))
            draw(search, bbox, "debug/{:06d}_s.jpg".format(index))

        # 以下部分还未参透
        cls, delta, delta_weight = self.anchor_target(self.anchors, bbox, self.size, neg)
        # 这个阶段不需要IoU计算，仅仅是取出数据，方便后面的训练；从上面的参数中也看出，没有传递need_iou这一参数，默认此参数的值是false
        if dataset.has_mask and not neg:  # 如果neg为true，则表示抽取的是一对负实例。
            mask_weight = cls.max(axis=0, keepdims=True)  # 如果keepdims=True，那么被减少的那个轴会以维度1保留在结果中
                                                        # 从而也印证了下面一行是第0个维度方向为什么会设置为1
        else:
            mask_weight = np.zeros([1, cls.shape[1], cls.shape[2]], dtype=np.float32)

        template, search = map(lambda x: np.transpose(x, (2, 0, 1)).astype(np.float32), [template, search])
        
        mask = (np.expand_dims(mask, axis=0) > 0.5) * 2 - 1  # 1*H*W

        return template, search, cls, delta, delta_weight, np.array(bbox, np.float32), \
               np.array(mask, np.float32), np.array(mask_weight, np.float32)

