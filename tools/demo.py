# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
from tools.test import *

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo') #解析器创建

 # metavar的作用是：如果你用--help查看帮助时，帮助信息将以--resume PATH的形式告诉你--resume后面要加上路径 你可以通过在terminal中
# python demo.py --help 的方式验证；
# 这是虽然定义的是optional argument ‘--resume’，但是由于required=True，所以是强制性的，必须要在执行脚本时声明
parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')

# 定义了欧平添了dest表示在变量解析后，原本名为‘--config’的变量将以名为‘config’的属性形式被引用
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='../../data/tennis', help='datasets')

# 这里也是定义可选变量，action=‘store_true’表示如果执行脚本时明确的地提出--cpu，则args.CPU=True，否则为False。这里的解引用会将
#可选变量名的前缀--去掉，并转为为大写字母，中间的'-'转变为底部'_'
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()

if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #大部分情况下，设置这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)
    from custom import Custom
    siammask = Custom(anchors=cfg['anchors']) # anchors从哪里获得？？？
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)  #eval属于切换到预测模式，并推送到GPU或CPU上运行

    # Parse Image file
    img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
    ims = [cv2.imread(imf) for imf in img_files]

    # Select ROI
    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        init_rect = cv2.selectROI('SiamMask', ims[0], False, False)
        x, y, w, h = init_rect # 返回来4个值
    except:
        exit()

    toc = 0
    for f, im in enumerate(ims):
        tic = cv2.getTickCount()
        if f == 0:  # init
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
        elif f > 0:  # tracking
            state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track
            location = state['ploygon'].flatten() # 这是干啥的？
            mask = state['mask'] > state['p'].seg_thr  # seg_thr是个啥？？？？？

            im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2] # 这一步玩的就是mask。。。
            cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            cv2.imshow('SiamMask', im)
            key = cv2.waitKey(1) # 持续1ms，如果中间有什么键按下，则返回该键所对应的ASCII码，肯定是正数；否则返回-1
            if key > 0:
                break

        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
