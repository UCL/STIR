#! /bin/sh
# non-robust script to make a .par file for generate_image from one for 
# list_ROI_values

if [ $# -ne 2 ]; then
  echo Usage: $0 output_generate_file input_ROIfile
  exit 1
fi

fitfile=$1
ROIfile=$2

cat <<EOF > $fitfile
generate_image Parameters :=
output filename:=ROI_image
X output image size (in pixels):=128
Y output image size (in pixels):=128
Z output image size (in pixels):=95
X voxel size (in mm):= 2.05941
Y voxel size (in mm):= 2.05941
Z voxel size (in mm) :=2.425

value := 1

EOF

startline=`grep -i -n "shape type" $ROIfile | head -n 1 | awk -F: '{ print $1 }'`
tail -n +$startline $ROIfile |grep -v -i "value :=" | \
   sed -e "s/ROI shape type/shape type/" >> $fitfile

echo edit $fitfile for pixel sizes etc
