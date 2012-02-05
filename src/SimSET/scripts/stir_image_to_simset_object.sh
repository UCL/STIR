#! /bin/bash 
#  $Id$
#
# This script attempts to find dimensions etc of an image using list_image_info
# and calls write_phg_image_info to write the corresponding lines for a 
# PHG input file to stdout.

#  Copyright (C) 2011-07-01 - $Date$, Kris Thielemans
#  This file is part of STIR.
#
#  This file is free software; you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation; either version 2.1 of the License, or
#  (at your option) any later version.

#  This file is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  See STIR/LICENSE.txt for details
#      
# Author: Kris Thielemans

print_usage_and_exit()
{
  echo "usage:"
  echo "   `basename $prog` image_filename"
  echo "This script attempts to write the object-spec corresponding to an image to stdout."
  exit 1
}

prog=$0
if [ $# -ne 1 ]; then
  print_usage_and_exit
  exit 1
fi

image=$1

if [ ! -r ${image} ]; then
    echo "File ${image} not found"
    exit 1
fi

set -e # exit on error
trap "echo ERROR in $prog $image" ERR

# Find image dimensions etc
first_edge_coords=(`list_image_info ${image}| awk -F: '/first edge/ {print $2}'|tr -d '{},' `)

if [ -z "${first_edge_coords}" ]; then
    echo "File ${image} not readable by STIR?"
    exit 1
fi

last_edge_coords=(`list_image_info ${image}| awk -F: '/last edge/ {print $2}'|tr -d '{},' `)
num_voxels=(`list_image_info ${image}| awk -F: '/Number of voxels/ {print $2}'|tr -d '{},' `)

xmin=${first_edge_coords[2]}
xmax=${last_edge_coords[2]}
ymin=${first_edge_coords[1]}
ymax=${last_edge_coords[1]}
zmin=${first_edge_coords[0]}
zmax=${last_edge_coords[0]}
numx=${num_voxels[2]}
numy=${num_voxels[1]}
numz=${num_voxels[0]}

write_phg_image_info $numz $numx $numy $xmin $xmax $ymin $ymax $zmin $zmax 
