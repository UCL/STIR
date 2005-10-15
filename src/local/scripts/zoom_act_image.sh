#
# $Id$
#
#
#  Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
#      See STIR/LICENSE.txt for details
#           
# Author /Charalampos Tsoumpas 07/10/2005

# This script is used to zoom the activity image, given the zoom factor. The image preserves the total counts. 

#! /bin/sh
#PBS -k eo 

if [ $# -ne 4 ]; then
    echo "usage: $0 output_image input_image.hv zoom_xy zoom_z \\"
    exit 1
fi
set -e
trap " echo ERROR in executing script" ERR

#echo "$USER give the projdata file"
output_image_filename=$1
input_image=$2
#echo "$USER give the zoom"
zoom_xy=$3 # 0.5
zoom_z=$4 # 0.5 for having 

voxels_x=`less ${input_image} | grep -F "matrix size [1]" |awk '{print $5}'`
voxels_y=`less ${input_image} | grep -F "matrix size [2]" |awk '{print $5}'`
voxels_z=`less ${input_image} | grep -F "matrix size [3]" |awk '{print $5}'`
new_voxels_xy=`echo ${voxels_x} ${voxels_y} ${zoom_xy} | awk ' { printf ("%.0f", ($1+$2)*$3*0.5)}'`  
new_voxels_z=`echo ${voxels_z} ${zoom_z} | awk ' { printf ("%.0f", $1*$2)}'` 
scale_att=`echo ${zoom_z} ${zoom_xy} | awk ' { printf ("%.5f", $1*$2*$2)}'` 
#voxel_size_z=`less ${input_image} | grep -F "scaling factor (mm/pixel) [3]" |awk '{print $6}'` 
#offset_z=`echo ${voxels_z} ${new_voxels_z} ${zoom_z} ${voxel_size_z} | awk ' { printf ("%.5f", -0.5*$4*($2/$3-$1))}'`
echo "voxels_x=${voxels_x}"
echo "voxels_y=${voxels_y}"
echo "voxels_z=${voxels_z}"
echo "new_voxels_xy=${new_voxels_xy}"
echo "new_voxels_z=${new_voxels_z}"
echo "offset_z=$offset_z"

zoom_image ${output_image_filename} ${input_image} ${new_voxels_xy} ${zoom_xy} 0 0 ${new_voxels_z} ${zoom_z} 0



