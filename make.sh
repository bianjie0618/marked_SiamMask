#!/usr/bin/env bash

cd utils/pyvotkit
python setup.py build_ext --inplace  # ���ؽ�����������һ������չ���ļ�������developer�������飬���Ӧ���û�Ҫ���ľ���install
cd ../../

cd utils/pysot/utils/
python setup.py build_ext --inplace
cd ../../../
