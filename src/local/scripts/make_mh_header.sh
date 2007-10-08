#! /bin/sh
# copy MetaIO header (normally called .mhd) for new target binary file

if [ $# -ne 3 ]; then
    echo "usage: $0 \\"
    echo "    Input_mh_file Target_bin_file Output_mh_file"
    echo "will make Output_mh_file that refers to Target_bin_file"
    exit 1
fi

set -e # exit on error


#echo "$USER give the Input_mh_file, Target_bin_file, Output_mh_hile"

Input_mh_file=$1
Target_bin_file=$2 
Output_mh_file=$3

if [ ! -r $Input_mh_file ]
then
    echo "Input file $Input_mh_file not found - Aborting"
    exit 1
fi

if [ ! -r $Target_bin_file ]
then
    echo "Input file $Target_bin_file not found - Aborting"
    exit 1
fi

previous_bin_file=`grep "ElementDataFile"  ${Input_mh_file} |awk 'BEGIN { FS="=" } {print $2}'`
echo "Replacing ${previous_bin_file} with ${Target_bin_file} in header"

sed -e "s/${previous_bin_file}/ ${Target_bin_file}/" <  ${Input_mh_file} > ${Output_mh_file}


