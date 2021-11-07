#!/bin/sh
# Create a frame definition file in the STIR fdef file format spanning all of the data
# This uses list_lm_events to find the first and last timing event.
# All data before the first timing event will be skipped.
#
#
#  Copyright (C) 2020, University College London
#  This file is part of STIR.
#
#  SPDX-License-Identifier: Apache-2.0
#
#  See STIR/LICENSE.txt for details
#
# Author: Kris Thielemans

if [ $# -ne 2 ]; then
  echo "Usage: `basename $0` output_filename listmode_filename"
  echo "This creates a STIR fdef file with 1 time frame covering the first to last timing event"
  exit 1
fi

out_filename=$1
list_filename=$2

echo "Finding first and last timing event in $list_filename (might take some time)"
first=`list_lm_events --num-events-to-list 1 "$list_filename"|awk '/Time/{print $2}'`
if [ -z "$first" ]; then
    echo "Error reading $list_filename" >&2
    exit 1
fi
last=`list_lm_events "$list_filename" |tail |grep Time |tail -n 1|awk '/Time/{print $2}'`
if [ -z "$last" ]; then
    echo "Error reading last event from $list_filename" >&2
    exit 1
fi
duration=`echo $first $last|awk '{print ($2 - $1)/1000}'`
first_secs=`echo $first |awk '{print $1/1000}'`
echo "Found first event: $first, last event: $last, duration in secs: $duration"
echo "0 $first_secs" > "$out_filename"
echo "1 $duration" >> "$out_filename"
