#! /bin/sh

if [ $# -ne 2 ]; then
  echo Usage: $0 output_ROIfile input_generate_file
  exit 1
fi

fitfile=$2
ROIfile=$1

cat <<EOF > $ROIfile
ROIValues Parameters :=
   number of samples to take for ROI template-z:=1
   number of samples to take for ROI template-y:=1
   number of samples to take for ROI template-x:=1
EOF

startline=`grep -i -n "shape type" $fitfile | head -n 1 | awk -F: '{ print $1 }'`
tail -n +$startline $fitfile |grep -v -i "value :=" | \
   sed -e "s/shape type/ROI shape type/" >> $ROIfile
