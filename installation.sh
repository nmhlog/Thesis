#!/bin/bash
apt-get update
pip install -r requirements.txt
apt-get install libsparsehash-dev
conda install -c bioconda google-sparsehash
conda install -y libboost

pip install spconv-cu113

cd lib/hais_ops/
python setup.py build_ext develop

cd lib/softgroup_ops/
python setup.py build_ext develop