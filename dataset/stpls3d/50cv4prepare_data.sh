#!/bin/bash
echo "down_sampling"
cp crossvalidation/50/50cv4prepare_data_inst_instance_stpls3d.py 50cv4prepare_data_inst_instance_stpls3d.py

python 50cv4prepare_data_inst_instance_stpls3d.py
python prepare_data_statistic_stpls3d.py --data_folder traincv4_50_50 > note_weighttraincv4_50_50.txt
python prepare_data_statistic_stpls3d.py --data_folder traincv4_50_25 > note_weighttraincv4_50_25.txt
rm 50cv4prepare_data_inst_instance_stpls3d.py

