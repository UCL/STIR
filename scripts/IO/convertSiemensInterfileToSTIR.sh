#! /bin/sh
# Preliminary script to convert Interfile headers from Siemens format to STIR
# WARNING: this currently doesn't check a load of things
# for instance, for images, x-y offset and orientation are not checked
#
# Author: Kris Thielemans
# Copyright: University College London

# check number of arguments
if [ $# -ne 2 ]; then
  echo "Usage: convertSiemensInterfileToSTIR.sh siemens-header-filename stir-header-filename"
  echo "Use ONLY for image headers."
  exit 1
fi

# give names to arguments
# we will use e.g. "${in}" below to get the corresponding values. The quotes serve to handle spaces in filenames.
in=$1
out=$2

# check if filenames are different
if [ "${in}" = "${out}" ]; then
  echo "Input and output file names need to be different"
  exit 1
fi

# check if input is readable
if [ -r "${in}" ]; then
  : # ok, it's readable
else
    echo "Input file is not readable"
  exit 1
fi

# check if it's a sinogram
fgrep -i "sinogram subheader" "${in}" >/dev/null
if [ $? = 0 ]; then
  # yes it is
  echo "STIR is currently able to parse Siemens Interfile format sinograms and norm files. This script ends now."
  exit 1
fi

# replace a number of keywords according to the proposed Interfile standard
# due to restrictions of many sed versions, we first use @ in the replacement string to indicate a newline, 
# and then use tr to replace @ with an actual newline 
sed \
 -e "s/GENERAL IMAGE DATA *:=/GENERAL IMAGE DATA :=@!type of data := PET@/" \
 -e "s/image data byte order/imagedata byte order/" \
 -e "s#scale factor (mm/pixel)#scaling factor (mm/pixel)#" \
 -e "s/data offset in bytes\[2\]:=/;data offset in bytes[2]:=/" \
 -e "s/!*image duration (sec)/number of time frames:=1@\
!image duration (sec)[1]/" \
 -e "s/\(image relative start time (sec)\)/\1[1]/" \
 "${in}" | tr @ "\n" > "${out}"

# check if sed worked
if [ $? -ne 0 ]; then
  echo 'sed command failed. Output file writable?'
  exit 1
fi

# append "END OF INTERFILE" as Siemens doesn't do it
echo "!END OF INTERFILE :=" >> "${out}"
