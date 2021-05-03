#!/bin/sh
# A simple script to print voxel sizes of an image
# Example:
# print_voxel_sizes.sh img.hv
# => will print voxel sizes (in mm) as 3 numbers separated by spaces, in order z y x
#
#  Copyright (C) 2011- 2011, Hammersmith Imanet Ltd
#  This file is part of STIR.
#
#  SPDX-License-Identifier: Apache-2.0
#
#  See STIR/LICENSE.txt for details
#      
# Author Kris Thielemans
# 
if [ $# -ne 1 ]; then
  echo "Usage: `basename $0` filename"
  echo "Prints voxel-sizes in order z y x"
  exit 1
fi

filename=$1

list_image_info "${filename}" |awk -F: '/[Vv]oxel-size/ {value=$2; gsub("[{},]"," ",value);print value}'
