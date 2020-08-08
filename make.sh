#!/usr/bin/env bash

cd utils/pyvotkit
python setup.py build_ext --inplace  # 本地建立“发布”一个“扩展”文件，这是developer做的事情，相对应的用户要做的就是install
cd ../../

cd utils/pysot/utils/
python setup.py build_ext --inplace
cd ../../../
