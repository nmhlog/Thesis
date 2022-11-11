#!/bin/bash
echo "down_sampling"
cp crossvalidation/50/50cv3prepare_data_inst_instance_stpls3d.py

python 50cv3prepare_data_inst_instance_stpls3d.py
python prepare_data_statistic_stpls3d.py --data_folder traincv3_50_50 > note_weighttraincv3_50_50.txt
python prepare_data_statistic_stpls3d.py --data_folder traincv3_50_25 > note_weighttraincv3_50_25.txt
rm 50cv3prepare_data_inst_instance_stpls3d.py