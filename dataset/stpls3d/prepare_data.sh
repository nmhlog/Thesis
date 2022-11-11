#!/bin/bash
echo "down_sampling"
python down_sampling.py
python prepare_data_statistic_stpls3d.py --data_folder traincv2_50_50 > note_weight100100.txt
python prepare_data_statistic_stpls3d.py --data_folder traincv2_50_25 > note_weight10050.txt


