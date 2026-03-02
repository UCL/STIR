#! /bin/bash
#
# This script attempts to make Interfile headers (.hv STIR style and 
# .ahv Analyze style) for a raw float image output by Simset.
# It uses the geometric info in a Simset PHG file.
# See the usage message below for some more info

#  Copyright (C) 2009-07-08, Hammersmith Imanet Ltd
#  Copyright (C) 2011-07-01 - 2012, Kris Thielemans
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
  echo "   `basename $prog` simset_phg_params_filename raw_float_filename"
  echo "This attempts to make Interfile headers (.hv STIR style and .ahv Analyze style)"
  echo "for a raw float image output by Simset by using the geometric info in a Simset PHG file."
  echo "WARNING: This does not work yet with different sized emission and attenuation images."
  echo "warning: check if output is ok. This is not very stable code...."
  exit 1
}

# find_param params_file param
find_param()
{
 # get all occurences and sort numerically
 grep "[[:blank:]]$2" $1|uniq|awk -F= '{print $2}'|sort -g
}

prog=$0
if [ $# != 2 ]; then
  print_usage_and_exit
fi

binary_file="$2"
params_file="$1"

if [ ! -r "$binary_file" ]; then
    echo "ERROR: $prog cannot read binary_file $binary_file" 1>&2 
  exit 1 
fi 
if [ ! -r "$params_file" ]; then
    echo "ERROR: $prog cannot read Simset params_file $params_file" 1>&2 
  exit 1 
fi 

set -e # exit on error
trap "echo ERROR in $prog $input_file" ERR

# get rid of some common extension in binary_file to construct header name
header_name=${binary_file%.f32}
header_name=${header_name%.dat}
output_file_hv=${header_name}.hv
output_file_ahv=${header_name}.ahv

# remove directory info from the binary_file variable such that we don't store it in the header
binary_file=`basename ${binary_file}`

xMin=`find_param $params_file xMin`
xMax=`find_param $params_file xMax`
yMin=`find_param $params_file yMin`
yMax=`find_param $params_file yMax`
zMin=`find_param $params_file zMin|head -n 1`
zMax=`find_param $params_file zMin|tail -n 1`
DimSize1=`find_param $params_file num_X_bins`
DimSize2=`find_param $params_file num_Y_bins`
DimSize3=$(( `find_param $params_file slice_number|tail -n 1` + 1 ))
ElementSpacing1=`python -c "print ($xMax - $xMin)*10/$DimSize1"`
ElementSpacing2=`python -c "print ($yMax - $yMin)*10/$DimSize2"`
ElementSpacing3=`python -c "print ($zMax - $zMin)*10/($DimSize3 - 1)"`
Offset1=`python -c "print $xMin*10 + $ElementSpacing1/2"`
Offset2=`python -c "print $yMin*10 + $ElementSpacing2/2"`
Offset3=`python -c "print $zMin*10"`

number_format="float"; bytes_per_pixel=4;
byte_order=`python -c 'import sys; print sys.byteorder'`endian
HeaderSize=0

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


echo "Created $output_file_hv and $output_file_ahv (assuming ${byte_order} byte order)"
