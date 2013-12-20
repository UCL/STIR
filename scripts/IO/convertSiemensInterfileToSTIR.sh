#! /bin/sh
# Preliminary script to convert Interfile headers from Siemens format to STIR
# WARNING: this currently doesn't check a load of things
# for instance, for images, x-y offset and orientation are not checked
# see notes below for sinograms
#
# Author: Kris Thielemans
# Copyright: University College London

# check number of arguments
if [ $# -ne 2 ]; then
  echo "Usage: convertSiemensInterfileToSTIR.sh siemens-header-filename stir-header-filename"
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

# check if it's a sinogram
fgrep -i "sinogram subheader" "${in}" >/dev/null
if [ $? = 0 ]; then
  # yes it is, so we need to do some more replacements
  mv "${out}" "${out}.tmp"
  # first find out if it's 3D or 2D Data by checking how many sinograms there are
  grep -i "%segment table *:= *{ *127 *}" "${in}" >/dev/null
  if [ $? = 0 ]; then
    # we found 127, so it's 2D
    # TODO could be larger ring diff, but STIR will currently ignore it anyway
    minringdiff='{-1}'
    maxringdiff='{+1}'
  else
    # TODO construct this based on actual max ring diff. also take axial compression into account
    # below is ok for mMR probably
    minringdiff='{ 0,1,-1,2,-2,3,-3,4,-4,5,-5,6,-6,7,-7,8,-8,9,-9,10,-10,11,-11,12,-12,13,-13,14,-14,15,-15,16,-16,17,-17,18,-18,19,-19,20,-20,21,-21,22,-22,23,-23,24,-24,25,-25,26,-26,27,-27,28,-28,29,-29,30,-30,31,-31,32,-32,33,-33,34,-34,35,-35,36,-36,37,-37,38,-38,39,-39,40,-40,41,-41,42,-42,43,-43,44,-44,45,-45,46,-46,47,-47,48,-48,49,-49,50,-50,51,-51,52,-52,53,-53,54,-54,55,-55,56,-56,57,-57,58,-58,59,-59,60,-60}'
    maxringdiff=${minringdiff}
  fi
  # TODO by-view/by-sino based on "plane/projection" order in matrix label
  # code below assumes it's by-sino, but this is incorrect for Siemens 2D data apparently
  sed \
 -e "s/number of dimensions:=3/number of dimensions:=4/" \
 -e "s/applied corrections:= *$/applied corrections:={None}/" \
 -e "s/applied corrections:=\(.*\)radial arc-correction\(.*\)$/applied corrections:=\1arc-correction\2/"\
 -e "s/matrix axis label\(.*\):=bin/matrix axis label\1:= tangential coordinate/" \
 -e "s/matrix axis label\(.*\):=x/matrix axis label\1:= tangential coordinate/" \
 -e "s/matrix axis label\(.*\):=projection/matrix axis label\1:=view/" \
 -e "s/matrix axis label\(.*\):=plane/matrix axis label\1:= axial coordinate/" \
 -e "s/matrix size *\[3\].*$//" \
 -e "s/%number of segments *:=/matrix axis label [4] := segment@matrix size [4] :=/" \
 -e "s/%segment table *:=\(.*\)/matrix size [3] :=\1@ minimum ring difference per segment := ${minringdiff}@\
 maximum ring difference per segment := ${maxringdiff}@\
/" \
   "${out}.tmp" | tr @ "\n" > "${out}"
   rm "${out}.tmp"
fi


