#!/bin/bash
echo "Analyst Points And Size"

python analyst_data_statistic_stpls3d.py --data_folder train_50_50 > statistic_analyst_50_50.txt
python analyst_data_statistic_stpls3d.py --data_folder train_50_25 > statistic_analyst_50_25.txt
python analyst_data_statistic_stpls3d.py --data_folder val > statistic_analyst_val_50_50.txt


