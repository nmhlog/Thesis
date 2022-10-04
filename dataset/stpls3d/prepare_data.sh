#!/bin/bash
echo Preprocess data
python prepare_data_inst_instance_stpls3d.py
python prepare_data_statistic_stpls3d.py --data_folder train_100_100 > note_weight100100.txt
python prepare_data_statistic_stpls3d.py --data_folder train_100_50 > note_weight10050.txt


