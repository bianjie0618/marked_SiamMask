# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from __future__ import division

import os
import logging
import sys
import math

if hasattr(sys, 'frozen'):  # support for py2exe
    _srcfile = "logging%s__init__%s" % (os.sep, __file__[-4:])
elif __file__[-4:].lower() in ['.pyc', '.pyo']:
    _srcfile = __file__[:-4] + '.py'
else:
    _srcfile = __file__   # __file__ ：当前文件路径
_srcfile = os.path.normcase(_srcfile) # 根据系统规范化路径名的大小写


logs = set()


class Filter:
    def __init__(self, flag):
        self.flag = flag

    def filter(self, x): return self.flag


class Dummy: # 人体模型，仿制品。
    def __init__(self, *arg, **kwargs):
        pass

    def __getattr__(self, arg):  # 把事情压下去，不作任何处理
        def dummy(*args, **kwargs): pass
        return dummy


def get_format(logger, level):
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID']) # 集群进程号ID; 先可以不理。。。

        if level == logging.INFO:
            logger.addFilter(Filter(rank == 0))
    else:
        rank = 0
    format_str = '[%(asctime)s-rk{}-%(filename)s#%(lineno)3d] %(message)s'.format(rank)
    formatter = logging.Formatter(format_str)
    return formatter


def get_format_custom(logger, level):
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        if level == logging.INFO:
            logger.addFilter(Filter(rank == 0))
    else:
        rank = 0
    format_str = '[%(asctime)s-rk{}-%(message)s'.format(rank)
    formatter = logging.Formatter(format_str)
    return formatter


def init_log(name, level = logging.INFO, format_func=get_format):
    if (name, level) in logs: return  # 实例化logger，加入StreamHandler处理器
    logs.add((name, level))
    logger = logging.getLogger(name) # 实例化
    logger.setLevel(level)
    ch = logging.StreamHandler() # 流处理器，输出到标准输出里
    ch.setLevel(level)
    formatter = format_func(logger, level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def add_file_handler(name, log_file, level = logging.INFO):
    logger = logging.getLogger(name)  # 实例化name="name"的logger，create if necessary;添加FileHandler处理器
    fh = logging.FileHandler(log_file) # 文件名log_file
    fh.setFormatter(get_format(logger, level))
    logger.addHandler(fh)


init_log('global')


def print_speed(i, i_time, n):
    """print_speed(index, index_time, total_iteration)"""
    logger = logging.getLogger('global')
    average_time = i_time
    remaining_time = (n - i) * average_time
    remaining_day = math.floor(remaining_time / 86400)
    remaining_hour = math.floor(remaining_time / 3600 - remaining_day * 24)
    remaining_min = math.floor(remaining_time / 60 - remaining_day * 1440 - remaining_hour * 60)
    logger.info('Progress: %d / %d [%d%%], Speed: %.3f s/iter, ETA %d:%02d:%02d (D:H:M)\n' % (i, n, i/n*100, average_time, remaining_day, remaining_hour, remaining_min))


# 不明白这个函数想要干个啥
def find_caller():
    def current_frame():
        try:
            raise Exception
        except:
            return sys.exc_info()[2].tb_frame.f_back  # 记录关于traceback的错误信息，可以参考下方链接
            # https://blog.csdn.net/deooou1234/article/details/86567090
    f = current_frame()
    if f is not None:
        f = f.f_back
    rv = "(unknown file)", 0, "(unknown function)" # rv变成了元组tuple
    while hasattr(f, "f_code"):  #
        co = f.f_code
        filename = os.path.normcase(co.co_filename)
        rv = (co.co_filename, f.f_lineno, co.co_name)
        if filename == _srcfile:
            f = f.f_back
            continue
        break
    rv = list(rv)
    rv[0] = os.path.basename(rv[0])
    return rv


class LogOnce:
    def __init__(self):
        self.logged = set()
        self.logger = init_log('log_once', format_func=get_format_custom)

    def log(self, strings):
        fn, lineno, caller = find_caller()
        key = (fn, lineno, caller, strings)
        if key in self.logged:
            return
        self.logged.add(key)
        message = "{filename:s}<{caller}>#{lineno:3d}] {strings}".format(filename=fn, lineno=lineno, strings=strings, caller=caller)
        self.logger.info(message)


once_logger = LogOnce()


def log_once(strings):
    once_logger.log(strings)
