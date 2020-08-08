# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import json
from os.path import exists


def proccess_loss(cfg):
    if 'reg' not in cfg: # 回归损失
        cfg['reg'] = {'loss': 'L1Loss'}
    else:
        if 'loss' not in cfg['reg']:
            cfg['reg']['loss'] = 'L1Loss'

    if 'cls' not in cfg: # 分类损失
        cfg['cls'] = {'split': True}

    cfg['weight'] = cfg.get('weight', [1, 1, 36])  # cls, reg, mask 获取key=weight对应的值，如果没有，就返回[1,1,36]


def add_default(conf, default):
    default.update(conf)
    return default


def load_config(args):
    assert exists(args.config), '"{}" not exists'.format(args.config) # 这一部可以检查是否真的有相关的配置文件
    config = json.load(open(args.config)) #打开cinfig.json文件，进行配置
    # deal with network
    if 'network' not in config:
        print('Warning: network lost in config. This will be error in next version')

        config['network'] = {}

        if not args.arch:
            # dic = {'arch':''}
            # if not dic['arch']:
            #     print('jianguile')
            #     print(not dic['arch']) # 你可以试试一个键所对应的vlaue如果是空值，not dic['arch'] 返回的就是True
            raise Exception('no arch provided')
    args.arch = config['network']['arch'] # arch到底是个啥？？？ arch就是个architecture name

    # deal with loss
    if 'loss' not in config:
        config['loss'] = {}

    proccess_loss(config['loss'])

    # deal with lr
    if 'lr' not in config:
        config['lr'] = {}
    default = {
            'feature_lr_mult': 1.0,
            'rpn_lr_mult': 1.0,
            'mask_lr_mult': 1.0,
            'type': 'log', # binary logistic regression
            'start_lr': 0.03
            }
    default.update(config['lr'])
    config['lr'] = default

    # clip
    if 'clip' in config or 'clip' in args.__dict__:  # 这种方法双下划线的方法是什么？？？
        if 'clip' not in config:
            config['clip'] = {}
        config['clip'] = add_default(config['clip'],
                {'feature': args.clip, 'rpn': args.clip, 'split': False})
        if config['clip']['feature'] != config['clip']['rpn']:
            config['clip']['split'] = True
        if not config['clip']['split']:
            args.clip = config['clip']['feature'] #

    return config

