#!/bin/bash
cd lib/hais_ops/ 
rm -r HAIS_OP.cpython-38-x86_64-linux-gnu.so
rm -r HAIS_OP.egg-info
rm -r build

export CPLUS_INCLUDE_PATH={conda_env_path}/hais/include:$CPLUS_INCLUDE_PATH
python setup.py build_ext develop

