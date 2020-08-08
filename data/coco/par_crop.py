# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------

# COCO如何使用以及如何只做数据，本案例已详细给出，后期要记录在github、csdn上
from pycocotools.coco import COCO
import cv2
import numpy as np
from os.path import join, isdir
from os import mkdir, makedirs
from concurrent import futures
import sys
import time
import argparse

parser = argparse.ArgumentParser(description='COCO Parallel Preprocessing for SiamMask')
parser.add_argument('--exemplar_size', type=int, default=127, help='size of exemplar')
parser.add_argument('--context_amount', type=float, default=0.5, help='context amount')
parser.add_argument('--search_size', type=int, default=511, help='size of cropped search region')
parser.add_argument('--enable_mask', action='store_true', help='whether crop mask')
parser.add_argument('--num_threads', type=int, default=24, help='number of threads')
args = parser.parse_args()


# Print iterations progress (thanks StackOverflow)
def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    # 这里要明确的是：在进性仿射变换时，已经将bbox的中心置于整个仿射变换的中心，即结果图片crop的中心就是bbox的中心
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]   # 仿射变换中的平移操作
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


def pos_s_2_bbox(pos, s):
    # 计算裁切的box的左上角和右下角的坐标位置(小数)
    return [pos[0]-s/2, pos[1]-s/2, pos[0]+s/2, pos[1]+s/2]


def crop_like_SiamFCx(image, bbox, exemplar_size=127, context_amount=0.5, search_size=255, padding=(0, 0, 0)):
    # 根据注释裁切图片，注释里有相关的box的位置和尺寸信息，间接确定出exemplar image的裁切尺寸，最终可以resize为127
    # 由于要进行训练，我们还需要进一步确定search_image上的裁切尺寸大小，最终确定resize为511并保存到crop511文件夹下
    target_pos = [(bbox[2]+bbox[0])/2., (bbox[3]+bbox[1])/2.]  # 是小数，不是整数
    target_size = [bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)  # s_z 就是我们想要划定的exemplar image上边框的大小，而这个大小并非127，但是最后会通过resize会变成127
    scale_z = exemplar_size / s_z   # scale_z就是resize的比例大小

    # 如果在exemplar_image上裁切的原始尺寸是s_z，那么在search_image上裁切的尺寸原始是多少呢？
    # 已经知道resize后输入神经网络的size：exemplar_size=127，search_size=511；还知道resize的比例大小是scale_z。一下三行就是计算公式
    d_search = (search_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), search_size, padding)
    return x


def crop_img(img, anns, set_crop_base_path, set_img_base_path,
             exemplar_size=127, context_amount=0.5, search_size=511, enable_mask=True):
    frame_crop_base_path = join(set_crop_base_path, img['file_name'].split('/')[-1].split('.')[0])  # 该行为止是最后的文件夹
    if not isdir(frame_crop_base_path): makedirs(frame_crop_base_path)

    im = cv2.imread('{}/{}'.format(set_img_base_path, img['file_name']))
    avg_chans = np.mean(im, axis=(0, 1))
    # 一个image_id会产生多个annotation id，即一张图片可以产生多个事物的segmentation
    # annotation的具体格式详见 http://cocodataset.org/#format-data
    for track_id, ann in enumerate(anns):   # 每个图片中有多个分割实例，每个分割实例在该图片中都有一个track_id，注意不是image_id
        rect = ann['bbox']
        if rect[2] <= 0 or rect[3] <= 0:   # rect[2]、rect[3]是h、w
            continue
        bbox = [rect[0], rect[1], rect[0]+rect[2]-1, rect[1]+rect[3]-1]

        x = crop_like_SiamFCx(im, bbox, exemplar_size=exemplar_size, context_amount=context_amount,  # this step need
                              search_size=search_size, padding=avg_chans)
        cv2.imwrite(join(frame_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(0, track_id)), x)

        if enable_mask:
            im_mask = coco.annToMask(ann).astype(np.float32)   # 转变成mask
            x = (crop_like_SiamFCx(im_mask, bbox, exemplar_size=exemplar_size, context_amount=context_amount,
                                   search_size=search_size) > 0.5).astype(np.uint8) * 255
            # 之所以上一句话有>0.5这个操作，是因为cv2.warpAffine的flags默认是线性插值cv2.INTER_LINEAR，会产生除0和1之外的中间数
            # 因此有必要就近归并到0或1
            cv2.imwrite(join(frame_crop_base_path, '{:06d}.{:02d}.m.png'.format(0, track_id)), x)


def main(exemplar_size=127, context_amount=0.5, search_size=511, enable_mask=True, num_threads=24):
    global coco  # will used for generate mask
    data_dir = '.'
    crop_path = './crop{:d}'.format(search_size)
    if not isdir(crop_path): mkdir(crop_path)

    for data_subset in ['val2017', 'train2017']:
        set_crop_base_path = join(crop_path, data_subset)
        set_img_base_path = join(data_dir, data_subset)

        anno_file = '{}/annotations/instances_{}.json'.format(data_dir, data_subset)

        # COCO api class that loads COCO annotation file and prepare data structures.
        # 建议看看源代码的这部分内容
        coco = COCO(anno_file)
        n_imgs = len(coco.imgs)
        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:   # 开启多线程操作
            fs = [executor.submit(crop_img, coco.loadImgs(id)[0],
                                  coco.loadAnns(coco.getAnnIds(imgIds=id, iscrowd=None)),  # 一张图可能有多个注释anns
                                  set_crop_base_path, set_img_base_path,
                                  exemplar_size, context_amount, search_size,
                                  enable_mask) for id in coco.imgs]
            for i, f in enumerate(futures.as_completed(fs)):
                printProgress(i, n_imgs, prefix=data_subset, suffix='Done ', barLength=40)
    print('done')


if __name__ == '__main__':
    since = time.time()
    main(args.exemplar_size, args.context_amount, args.search_size, args.enable_mask, args.num_threads)
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
