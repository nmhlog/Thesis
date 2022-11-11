#!/bin/bash
echo "down_sampling"
python prepare_data_inst_instance_stpls3d.py
python prepare_data_statistic_stpls3d.py --data_folder traincv2_50_50 > note_weighttraincv2_50_50.txt
python prepare_data_statistic_stpls3d.py --data_folder traincv2_50_25 > note_weighttraincv2_50_25.txt


