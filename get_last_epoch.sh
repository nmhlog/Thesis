dir_name=$1
num_epoch=$(ls $1 |grep epoch | cut -d _ -f 2|cut -d . -f 1|sort -n |tail -n 1)
echo "$1epoch_$num_epoch.pth"

