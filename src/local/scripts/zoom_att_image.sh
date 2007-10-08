#
# $Id$
#
#
#  Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
#      See STIR/LICENSE.txt for details
#      
# Author /Charalampos Tsoumpas 07/10/2005

# This script is used to zoom the attenuation image, given the zoom factor. The image preserves the mu factor using stir-math. 

#! /bin/sh
#PBS -k eo 


if [ $# -lt 4 -o $# -gt 5 ]; then
    echo "usage: $0 output_image input_image.hv zoom_xy zoom_z [ offset_z]" 1>&2
    exit 1 
fi
set -e
trap " echo ERROR in executing zoom_att_image.sh 1>&2" ERR

output_image_filename=$1
input_image=$2
zoom_xy=$3
zoom_z=$4
# offset_z default to 0
offset_z=${5:-0}

voxel_sizes=`get_image_dimensions.sh ${input_image}`
voxels_x=`echo ${voxel_sizes} |awk '{print $1}'`
voxels_y=`echo ${voxel_sizes} |awk '{print $2}'`
if [ $voxels_x != $voxels_y ] ; then 
echo "The voxels in the x and y dimensions are different. Cannot zoom...  " 
exit 1 
fi
voxels_xy=${voxels_x}
voxels_z=`echo ${voxel_sizes} |awk '{print $3}'`
# make sure that new voxel number is odd
new_voxels_xy=`echo ${voxels_xy} ${zoom_xy} | awk ' {  a=int($1*$2+.999); if (a%2==0) ++a; print a }'`
new_voxels_z=`echo ${voxels_z} ${zoom_z} | awk ' { printf ("%.0f", $1*$2)}'` 
scale_att=`echo ${zoom_z} ${zoom_xy} | awk ' { printf ("%.5f", $1*$2*$2)}'` 

#offset_z=`echo ${voxels_z} ${new_voxels_z} ${zoom_z} ${voxel_size_z} | awk ' { printf ("%.5f", -0.5*$4*($2/$3-$1))}'`
  
echo "voxels_x= ${voxels_x}"
echo "voxels_y= ${voxels_y}"
echo "voxels_z= ${voxels_z}"
echo "new_voxels_xy= ${new_voxels_xy}"
echo "new_voxels_z= ${new_voxels_z}"
echo "offset_z= $offset_z"

zoom_image ${output_image_filename} ${input_image} ${new_voxels_xy} ${zoom_xy} 0 0 ${new_voxels_z} ${zoom_z} ${offset_z}
stir_math --accumulate --times-scalar ${scale_att}  --including-first ${output_image_filename}.hv


