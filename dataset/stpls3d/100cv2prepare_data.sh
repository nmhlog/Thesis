#!/bin/bash
echo "down_sampling"
cp crossvalidation/100/100cv2prepare_data_inst_instance_stpls3d.py

python 100cv2prepare_data_inst_instance_stpls3d.py
python prepare_data_statistic_stpls3d.py --data_folder traincv2_100_100 > note_weighttraincv2_100_100.txt
python prepare_data_statistic_stpls3d.py --data_folder traincv2_100_50 > note_weighttraincv2_100_50.txt
rm 100cv2prepare_data_inst_instance_stpls3d.py