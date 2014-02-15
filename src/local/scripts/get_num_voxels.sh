#! /bin/sh
# $Id: get_image_dimensions.sh,v 1.2 2007-10-08 20:48:34 kris Exp $
# Attempts to find image dimensions (i.e. number of pixels in x,y,z)
# Kris Thielemans

if [ $# -ne 1 ]; then 
  echo "Usage: get_num_voxels.sh filename" 1>&2
  echo 'Prints image dimensions (number of pixels) as "x y z"' 1>&2
  exit 1
fi

list_image_info $1| awk -F: '/Number of voxels/ {print $2}'|tr -d '{}'|awk -F, '{print $3, $2, $1}'

