#! /bin/sh

if [ $# -ne 3 ]; then
    echo "usage: $0 \\"
    echo "    Input_hv_file Target_bin_file Output_hv_file"
    echo "will make Output_hv_file that refers to Target_bin_file"
    exit 1
fi

set -e # exit on error


#echo "$USER give the Input_hv_file, Target_bin_file, Output_hv_hile"

Input_hv_file=$1
Target_bin_file=$2 
Output_hv_file=$3

if [ ! -r $Input_hv_file ]
then
    echo "Input file $Input_hv_file not found - Aborting"
fi

if [ ! -r $Target_bin_file ]
then
    echo "Input file $Target_bin_file not found - Aborting"
fi

previous_bin_file=`grep "name of data file"  ${Input_hv_file} |awk 'BEGIN { FS=":=" } {print $2}'`
echo "Replacing ${previous_bin_file} with ${Target_bin_file}"

sed -e "s/${previous_bin_file}/ ${Target_bin_file}/" <  ${Input_hv_file} > ${Output_hv_file}


