#!/bin/sh
file=$1
rev=$2
replace=$3

previous_line=`~/devel/hgroot/find_dateline_of_rev.sh $file $rev`
# check if revision exists
if [ -z "${previous_line}" ]; then
  exit
fi

if [ ! -r $file.org ]; then
  cp -p $file $file.org 
fi
mv $file $file.temp
sed -e  "s/^${previous_line}$/${replace}/" $file.temp > $file
rm $file.temp
