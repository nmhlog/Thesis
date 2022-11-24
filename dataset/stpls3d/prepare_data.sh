#!/bin/bash
echo "Analyst Points And Size"

python analyst_data_statistic_stpls3d.py --data_folder traincv2_50_50 > statistic_analyst_50_50.txt
python analyst_data_statistic_stpls3d.py --data_folder traincv2_50_25 > statistic_analyst_50_25.txt


