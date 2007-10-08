#! /bin/bash
#  $Id$
# Author: Kris Thielemans

print_usage_and_exit()
{
  echo "usage: $prog file.mhd"
  echo "This attempts to make Interfile headers (.hv STIR style and .ahv Analyze style)"
  echo " for a MetaIO (.mhd) file as output by ITK."
  echo "warning: it is bound to fail on hand-crafted MetaIO files."
  echo "warning: it does NOT work on mha files (where the data follows the header)"
  exit 1
}

prog=$0
if [ $# != 1 ]; then
  print_usage_and_exit
fi

input_file="$1"
output_file_hv=${input_file%mhd}hv
output_file_ahv=${input_file%mhd}ahv

if [ ! -r $input_file ]; then
    echo "ERROR: $prog cannot read input_file $input_file" 1>&2 
  exit 1 
fi 

set -e # exit on error
trap "echo ERROR in $prog $input_file" ERR

if [ "`is_MetaIO.sh ${input_file}`" = 0 ]; then
  echo "ERROR: $prog expects a MetaIO file as input_file" 1>&2
  exit 1
fi

. MetaIOfunctions.sh

binary_file=`MetaIOfield ${input_file} ElementDataFile`
BinaryDataByteOrderMSB=`MetaIOfield ${input_file} BinaryDataByteOrderMSB`
ElementSpacing=`get_voxel_sizes.sh ${input_file}`
DimSize=`get_image_dimensions.sh ${input_file}`
Offset=`get_image_first_voxel_offset.sh ${input_file}`
ElementType=`MetaIOfield ${input_file} ElementType`
HeaderSize=`MetaIOfield ${input_file} HeaderSize`
if [ "$HeaderSize" = "" ]; then HeaderSize=0; fi
if [ $HeaderSize = -1 ]; then 
    echo "ERROR: $prog cannot handle HeaderSize=-1" 1>&2 
  exit 1 
fi 

DimSize1=`echo $DimSize | awk '{print $1}'`
DimSize2=`echo $DimSize | awk '{print $2}'`
DimSize3=`echo $DimSize | awk '{print $3}'`
Offset1=`echo $Offset | awk '{print $1}'`
Offset2=`echo $Offset | awk '{print $2}'`
Offset3=`echo $Offset | awk '{print $3}'`
ElementSpacing1=`echo $ElementSpacing | awk '{print $1}'`
ElementSpacing2=`echo $ElementSpacing | awk '{print $2}'`
ElementSpacing3=`echo $ElementSpacing | awk '{print $3}'`

case $ElementType in
   MET_SHORT) number_format="signed integer"; bytes_per_pixel=2;;
   MET_USHORT) number_format="unsigned integer"; bytes_per_pixel=2;;
   MET_FLOAT) number_format="float"; bytes_per_pixel=4;;
   *) echo "Unsupported ElementType $ElementType" 1>&2; exit 1;;
esac

case $BinaryDataByteOrderMSB in
  [fF]alse) byte_order=LITTLEENDIAN;;
  [tT]rue) byte_order=BIGENDIAN;;
  *)  echo "Unsupported BinaryDataByteOrderMSB $BinaryDataByteOrderMSB" 1>&2; exit 1;;
esac

cat > $output_file_hv <<EOF
!INTERFILE  :=
!name of data file := $binary_file
!imagedata byte order := $byte_order
type of data := PET
!number format := $number_format
!number of bytes per pixel := $bytes_per_pixel
number of dimensions := 3
matrix axis label [1] := x
!matrix size [1] := $DimSize1
scaling factor (mm/pixel) [1] := $ElementSpacing1
matrix axis label [2] := y
!matrix size [2] := $DimSize2
scaling factor (mm/pixel) [2] := $ElementSpacing2
matrix axis label [3] := z
!matrix size [3] := $DimSize3
scaling factor (mm/pixel) [3] := $ElementSpacing3
first pixel offset (mm) [1] := $Offset1
first pixel offset (mm) [2] := $Offset2
first pixel offset (mm) [3] := $Offset3
;number of time frames := 1
!data offset in bytes[1] := $HeaderSize
!END OF INTERFILE :=
EOF

cat > $output_file_ahv <<EOF
!INTERFILE  :=
!name of data file := $binary_file
!imagedata byte order := $byte_order
!number format := $number_format
!number of bytes per pixel := $bytes_per_pixel
number of dimensions := 2
!total number of images := $DimSize3
matrix axis label [1] := x
!matrix size [1] := $DimSize1
scaling factor (mm/pixel) [1] := $ElementSpacing1
matrix axis label [2] := y
!matrix size [2] := $DimSize2
scaling factor (mm/pixel) [2] := $ElementSpacing2
;Correct value is of keyword (commented out), but needs to be computed
;!slice thickness (pixels) := $ElementSpacing3 /$ElementSpacing1
;Value for Analyze
!slice thickness (pixels) := $ElementSpacing3
data offset in bytes := $HeaderSize
!END OF INTERFILE :=
EOF
