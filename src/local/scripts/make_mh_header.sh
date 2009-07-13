#! /bin/sh
# copy MetaIO header (normally called .mhd) for new target binary file
# This works by replacing the value of the ElementDataFile in the template header
# Author: Kris Thielemans

if [ $# -ne 3 ]; then
    echo "usage: $0 \\"
    echo "    Input_mh_file Target_bin_file Output_mh_file"
    echo "will make Output_mh_file that refers to Target_bin_file"
    exit 1
fi

set -e # exit on error

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

# remove directory info from target_bin_file if it's the same as for the header
Target_bin_dir=${Target_bin_file%/*}
Output_mh_dir=${Output_mh_file%/*}
if [ "$Output_mh_dir" == "$Target_bin_dir" ]; then
  Target_bin_file=${Target_bin_file##*/}
fi

# find bin_file in original header
previous_bin_file=`grep "ElementDataFile"  ${Input_mh_file} |awk 'BEGIN { FS="=" } {print $2}'`
echo "Making $Output_mh_file from $Input_mh_file, replacing ${previous_bin_file} with ${Target_bin_file} in header"

# now create Output_mh_file by using sed
# warning: next line will fail if the filenames contain a #
sed -e "s#${previous_bin_file}# ${Target_bin_file}#"  ${Input_mh_file} > ${Output_mh_file}


