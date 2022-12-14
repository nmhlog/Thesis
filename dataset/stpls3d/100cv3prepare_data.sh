#!/bin/bash
echo "down_sampling"
cp crossvalidation/100/100cv3prepare_data_inst_instance_stpls3d.py 100cv3prepare_data_inst_instance_stpls3d.py

python 100cv3prepare_data_inst_instance_stpls3d.py
python prepare_data_statistic_stpls3d.py --data_folder traincv3_100_100 > note_weighttraincv3_100_100.txt
python prepare_data_statistic_stpls3d.py --data_folder traincv3_100_50 > note_weighttraincv3_100_50.txt
rm 100cv3prepare_data_inst_instance_stpls3d.py
python down_sampling.py
zip -r data_cv3_100.zip val_cv3_100/ val_cv3_100_sampling/ traincv3_100_100/ traincv3_100_50/ note_weighttraincv3_100_50.txt note_weighttraincv3_100_100.txt