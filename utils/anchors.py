# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import numpy as np
import math
from utils.bbox_helper import center2corner, corner2center
# 该module已经全部看懂

class Anchors:
    def __init__(self, cfg):
        self.stride = 8  # 这个通过论文的table 8就可以发现，RoW(response of candidate window)确实是8
        self.ratios = [0.33, 0.5, 1, 2, 3]  # 其实每一个Row的中心都会产生这么多的aspect ratios的anchor
        self.scales = [8]  # 这个尺度放大8倍是指相对于一个步长跨度的面积为64的框，再放大8倍作为基本的anchor尺寸面积大小
        self.round_dight = 0
        self.image_center = 0  # 中心用一个数字的标量表示，说明长宽相等
        self.size = 0
        self.anchor_density = 1   # 密度很有意思，这里的密度是说在每一个步长方向上，要设立多少个锚点(要区别于anchor)。我们这里默认是1

        self.__dict__.update(cfg) # update是builtin函数，用来更新字典

        self.anchor_num = len(self.scales) * len(self.ratios) * (self.anchor_density**2)
        self.anchors = None  # in single position (anchor_num*4) 4表示x,y,w,h
        self.all_anchors = None  # in all position 2*(4*anchor_num*h*w)
        self.generate_anchors()

    def generate_anchors(self):
        # 这个方法是在in single position产生anchors的方法
        self.anchors = np.zeros((self.anchor_num, 4), dtype=np.float32)

        size = self.stride * self.stride        # size就是横纵方向一个步长的面积；注意，这个方法里面的size是私有的，
        count = 0                               # 不会被该类中的其他方法使用
        anchors_offset = self.stride / self.anchor_density
        anchors_offset = np.arange(self.anchor_density)*anchors_offset
        anchors_offset = anchors_offset - np.mean(anchors_offset)
        x_offsets, y_offsets = np.meshgrid(anchors_offset, anchors_offset)

        # x_offset, y_offset由一个RoW产生的anchors里面的每一个anchor的中心点的横纵坐标
        for x_offset, y_offset in zip(x_offsets.flatten(), y_offsets.flatten()):
            for r in self.ratios:                                           # 横纵比一循环
                if self.round_dight > 0:
                    ws = round(math.sqrt(size*1. / r), self.round_dight)  # 严格说，r是纵横比，即r=h/w
                    hs = round(ws * r, self.round_dight)
                else:
                    ws = int(math.sqrt(size*1. / r))
                    hs = int(ws * r)

                for s in self.scales:                                       # 尺度一循环
                    w = ws * s
                    h = hs * s
                    self.anchors[count][:] = [-w*0.5+x_offset, -h*0.5+y_offset, w*0.5+x_offset, h*0.5+y_offset][:]
                    count += 1                  # 第一二个是左上角的横纵坐标，第三四个是右下角的横纵坐标

    def generate_all_anchors(self, im_c, size):
        if self.image_center == im_c and self.size == size:
            return False
        self.image_center = im_c
        self.size = size         # size到底代表啥

        # 下面的一行是说：size是search image经过孪生网络后、再经过depth-wise的xcorr后的feature map的大小(17x17)，从论文中可以看出stride=16(相当于经过四次下采样)
        # 而feature map左上角 第一个点 对应的在输入孪生网络前的search image里的横/纵坐标值就是下面的公式计算的值
        # 而计算出来的a0x就是一个锚点，一个锚点就可以产生一系列的anchors
        a0x = im_c - size // 2 * self.stride
        ori = np.array([a0x] * 4, dtype=np.float32)
        zero_anchors = self.anchors + ori  # 这里的加法存在广播机制; generate_anchors里面产生的是相对坐标，作为中心是0；
                                          # 这里我们要给出锚点的绝对中心，才能得到绝对坐标
        x1 = zero_anchors[:, 0]
        y1 = zero_anchors[:, 1]
        x2 = zero_anchors[:, 2]
        y2 = zero_anchors[:, 3]

        x1, y1, x2, y2 = map(lambda x: x.reshape(self.anchor_num, 1, 1), [x1, y1, x2, y2])
        cx, cy, w, h = corner2center([x1, y1, x2, y2])

        disp_x = np.arange(0, size).reshape(1, 1, -1) * self.stride
        disp_y = np.arange(0, size).reshape(1, -1, 1) * self.stride

        cx = cx + disp_x
        cy = cy + disp_y

        # broadcast
        zero = np.zeros((self.anchor_num, size, size), dtype=np.float32)
        # 从这里我们可以看到，zero的第零个维度方向代表的是anchor的种类(种类包括不同的横纵比，尺度scale，密度)
        # 第一、二个维度方向代表的是每个锚点(即feature map点，要区别于anchor)针对于第零维度上某一类anchor的具体的横纵坐标值
        cx, cy, w, h = map(lambda x: x + zero, [cx, cy, w, h])  # 这四个值的每一个都需要一个shape=(self.anchor_num, size, size)的ndarray来描述
        x1, y1, x2, y2 = center2corner([cx, cy, w, h])

        self.all_anchors = np.stack([x1, y1, x2, y2]), np.stack([cx, cy, w, h])  # np.stack就是摞起来了，多了一个维度方向；注意和np.cat的区别;
        return True                                                             # 返回来的是tuple，第一个元素是两坐标表示法的anchor信息；第二个是中心做标加h,w


# if __name__ == '__main__':
#     anchors = Anchors(cfg={'stride':16, 'anchor_density': 2})
#     anchors.generate_all_anchors(im_c=255//2, size=(255-127)//16+1+8)
#     a = 1

