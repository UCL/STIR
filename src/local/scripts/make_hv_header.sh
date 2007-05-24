#! /bin/bash

if [ $# -ne 3 -a $# -ne 2 ]; then
    echo "usage: $0 \\" 1>&2
    echo "    Input_hv_file Target_bin_file [Output_hv_file]" 1>&2
    echo "will make Output_hv_file that refers to Target_bin_file" 1>&2
    echo "By default, Output_hv_file is constructed by replacing the" 1>&2
    echo "extension in Target_bin_file to that in Input_hv_file." 1>&2
    exit 1
fi

set -e # exit on error


#echo "$USER give the Input_hv_file, Target_bin_file, Output_hv_hile"

Input_hv_file=$1
Target_bin_file=$2 


if [ ! -r $Input_hv_file ]
then
    echo "Input file $Input_hv_file not found - Aborting"
    exit 1
fi

if [ ! -r $Target_bin_file ]
then
    echo "Input file $Target_bin_file not found - Aborting"
    exit 1
fi

if [ $# == 3 ]; then
  Output_hv_file=$3
else
  Output_hv_file=${Target_bin_file%.*}.${Input_hv_file#*.}
fi


previous_bin_file=`grep "name of data file"  ${Input_hv_file} |awk 'BEGIN { FS=":=" } {print $2}'`
echo "Replacing ${previous_bin_file} with ${Target_bin_file} in header ${Output_hv_file}"

sed -e "s/${previous_bin_file}/ ${Target_bin_file}/" <  ${Input_hv_file} > ${Output_hv_file}


