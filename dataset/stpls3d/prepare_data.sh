#!/bin/bash
echo Preprocess data
python dataset/stpls3d/down_sampling.py
python prepare_data_statistic_stpls3d.py --data_folder train_100_100 > note_weight100100.txt
python prepare_data_statistic_stpls3d.py --data_folder train_100_50 > note_weight10050.txt


