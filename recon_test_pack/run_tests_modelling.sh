
#! /bin/bash
#
#  Copyright (C) 2009 - 2011, Hammersmith Imanet Ltd
#  Copyright (C) 2013 - 2014, 2020 University College London
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

# A script to check internal consistency of the parametric imaging code.
# It tests both the indirect and direct methods.
# First, dynamic data is simulated from parametric images.
# Then the estimated parametric images are compared with the originals.
# 
# This script currently needs to be executed in "recon_test_pack"
# (This could be changed by adjusting INPUTDIR below).

# Author: Charalampos Tsoumpas
# Author: Kris Thielemans

# we do the next line such that the script aborts at any error (i.e. any non-zero return value of a command).
# Note that it only works in bash. all the rest will work in sh (except for a substition line, look for //)
set -e
trap "echo ERROR" ERR
#Run Parametric Reconstruction
WORKSPACE=`pwd`

INPUTDIR=$WORKSPACE/test_modelling_input/
NUMSUBS=4 # 16 subsets create a difference in the direct method of more than 8%!!!
ITER=40
SAVITER=40
export MAXSEG=1
export INPUTDIR
export NUMSUBS
export ITER
export MAXSEG
export SAVITER

#
# Options
#
MPIRUN=""

#
# Parse option arguments (--)
# Note that the -- is required to suppress interpretation of $1 as options 
# to expr
#
while test `expr -- "$1" : "--.*"` -gt 0
do

  if test "$1" = "--mpicmd"
  then
    MPIRUN="$2"
    shift 1
  elif test "$1" = "--help"
  then
    echo "Usage: `basename $0` [--mpicmd somecmd] [install_dir]"
    echo "(where [] means that an argument is optional)"
    #echo "See README.txt for more info."
    exit 1
  else
    echo Warning: Unknown option "$1"
    echo rerun with --help for more info.
    exit 1
  fi

  shift 1

done 

if [ $# -eq 1 ]; then
  echo "Prepending $1 to your PATH for the duration of this script."
  PATH=$1:$PATH
fi

# script might be broken if you set this to true
ECAT7=false

command -v generate_image >/dev/null 2>&1 || { echo "generate_image not found or not executable. Aborting." >&2; exit 1; }
command -v get_dynamic_images_from_parametric_images >/dev/null 2>&1 || { echo "get_dynamic_images_from_parametric_images not found or not executable. Aborting." >&2; exit 1; }
if $ECAT7 ; then
    imgext=img
    command -v conv_to_ecat7 >/dev/null 2>&1 || { echo "conv_to_ecat7 not found or not executable. Aborting." >&2; exit 1; }
    function assemble_images()
    {
        conv_to_ecat7 $*  "ECAT 931"
    }
else
    imgext=multihv
    function assemble_images()
    {
        create_multi_header $*
    }
fi

echo "Using executables like the following"
echo "Using `command -v get_dynamic_images_from_parametric_images`"
echo "Using `command -v forward_project`"

mkdir -p test_modelling_output
cd test_modelling_output
rm -f *img

cp ${INPUTDIR}time.fdef .
cp ${INPUTDIR}plasma.if .


# Create images
# IF NOT EXIST
if [ ! -r p0005.hv ]; then
generate_image ${INPUTDIR}generate_p0005.par
fi
if [ ! -r p5.hv ]; then
generate_image ${INPUTDIR}generate_p5.par
fi
assemble_images p0005-p5.${imgext} p0005.hv p5.hv
# test
extract_single_images_from_parametric_image p0005-p5_%d.hv p0005-p5.${imgext}
if compare_image p0005-p5_1.hv p0005.hv; then
    : # ok
else
    echo "Error creating parametric image"; exit 1
fi
if compare_image p0005-p5_2.hv p5.hv; then
    : # ok
else
    echo "Error creating parametric image"; exit 1
fi

# create dynamics

get_dynamic_images_from_parametric_images dyn_from_p0005-p5.hv p0005-p5.${imgext}  ${INPUTDIR}PatlakPlot.par

# run and test Patlak estimation
apply_patlak_to_images indirect_Patlak.hv dyn_from_p0005-p5.hv ${INPUTDIR}PatlakPlot.par
echo "Test Patlak round-trip"
extract_single_images_from_parametric_image indirect_Patlak_img_%d.hv indirect_Patlak.hv
echo "indirect to original"
for par in 1 2; do
    compare_image indirect_Patlak_img_${par}.hv p0005-p5_${par}.hv
done

# Create the appropriate proj_data files
extract_single_images_from_dynamic_image dyn_from_p0005-p5_img_f%dg1d0b0.hv dyn_from_p0005-p5.hv
# if [ ! -r fwd_dyn_from_p0005-p5.S ]; then

for fr in `count 23 28`; do
    forward_project fwd_dyn_from_p0005-p5_f${fr}g1d0b0  dyn_from_p0005-p5_img_f${fr}g1d0b0.hv ${INPUTDIR}ECAT_931_projdata_template.hs > fwd_dyn_from_p0005-p5_f${fr}g1d0b0.log 2>&1
done
#fi

# create "fake" normalisation file just for testing.
# we will use a sinogram full of ones (which won't influence the result)
stir_math -s --including-first --times-scalar 0 --add-scalar 1 all_ones.hs fwd_dyn_from_p0005-p5_f23g1d0b0.hs

tmpvar="" ;
for fr in `count 1 23 `; do
 tmpvar="$tmpvar fwd_dyn_from_p0005-p5_f23g1d0b0.hs"
done
for fr in `count 24 28 `; do
 tmpvar="$tmpvar fwd_dyn_from_p0005-p5_f${fr}g1d0b0.hs"
done

#tmpvar=`count --pre  "fwd_dyn_from_p0005-p5_f" --post "g1d0b0.hs" 1 28`

# create dynamic proj_data (by using the Multi file-format)
create_multi_header fwd_dyn_from_p0005-p5.hs $tmpvar

for direct in OSMAPOSL OSSPS ; do 
    cp ${INPUTDIR}P${direct}.par .
    echo "Test the direct P${direct} Patlak Plot reconstruction"
    rm -f P${direct}.log;
    INPUT=fwd_dyn_from_p0005-p5.hs INIT=indirect_Patlak.hv ${MPIRUN} P${direct} P${direct}.par > P${direct}.log 2>&1

    echo "Compare the direct parametric images to the original ones"
    extract_single_images_from_parametric_image p0005-p5_img_f%dg1d0b0.hv p0005-p5.${imgext}
    extract_single_images_from_parametric_image P${direct}_${ITER}_img_f%dg1d0b0.hv P${direct}_${ITER}.hv
    for par in 1 2; do
        compare_image -t .01 P${direct}_${ITER}_img_f${par}g1d0b0.hv p0005-p5_img_f${par}g1d0b0.hv  
    done
    echo "Comparison is OK"

done # POSMAPOSL POSSPS#

echo "Test the utility: 'mult_model_with_dyn_images'"
echo "Multiply the  dynamic images with the model matrix to get images in the parametric space."
# first make a copy
stir_math --parametric  test_mult_dyn_with_model.hv indirect_Patlak.hv
mult_model_with_dyn_images test_mult_dyn_with_model.hv dyn_from_p0005-p5.hv ${INPUTDIR}PatlakPlot.par
extract_single_images_from_parametric_image test_mult_dyn_with_model_img_f%dg1d0b0.hv test_mult_dyn_with_model.hv

echo " "
echo "Min Counts for Par 1 "

value_1=`list_image_info --min test_mult_dyn_with_model_img_f1g1d0b0.hv | awk '/min/ {print $3}' `
is_differ=`echo ${value_1} | awk ' { print ($1>.0001) } '` 
if [  ${is_differ} -eq 1 ]; then   
    echo "When estimate min_counts_in_images values do not match. Check mult_model_with_dyn_images.cxx"
    exit 1
else
    echo " Test OK "
fi

echo "Min Counts for Par 2 "
value_2=`list_image_info --min test_mult_dyn_with_model_img_f2g1d0b0.hv | awk '/min/ {print $3}' `
is_differ=`echo ${value_2} | awk ' { print ($1>.0001) } '` 
if [  ${is_differ} -eq 1 ]; then   
    echo "When estimate min_counts_in_images values do not match. Check mult_model_with_dyn_images.cxx"
    exit 1
else 
    echo " Test OK "
fi

echo "Max Counts for Par 1 "
value_3=`list_image_info --max test_mult_dyn_with_model_img_f1g1d0b0.hv | awk '/max/ {print $3}' `
# 6.391818 is the scale due to the zoom factor.
is_differ=`echo ${value_3} | awk ' { print (($1-1619.32*6.391818*6.391818)*($1-1619.32*6.391818*6.391818)>.5*6.391818*6.391818*6.391818*6.391818*.5) } '` 
if [  ${is_differ} -eq 1 ]; then   
    echo "When estimate max_counts_in_images values do not match. Check mult_model_with_dyn_images.cxx"
    exit 1
else
 echo " Test OK "
fi

echo "Max Counts for Par 2 "
value_4=`list_image_info --max test_mult_dyn_with_model_img_f2g1d0b0.hv | awk '/max/ {print $3}' `
is_differ=`echo ${value_4} | awk ' { print (($1-0.356552*6.391818*6.391818)*($1-0.356552*6.391818*6.391818)>.001*6.391818*6.391818*.001) } '` 
if [  ${is_differ} -eq 1 ]; then   
    echo "When estimate max_counts_in_images values do not match. Check mult_model_with_dyn_images.cxx"
    exit 1
else 
    echo " Test OK "
fi

# These lines are commented out. KT has no idea what they were supposed to test
# (there isn't actually any real test here currently)
## Extract Sensitivity Frames
#extract_single_images_from_dynamic_image sens_img_f%dg1d0b0.hv sens.hv
#
#echo "The sensitivity image is not tested, yet!!!"
## Multiply the parametric images with the sensitivity images ### NOT IMPLEMENTED YET
#for it in ${ITER}; do
#    for fr in `count 23 28` ; do
#	stir_math --mult mult_${it}_f${fr} dyn_sens_img_f${fr}g1d0b0.hv dyn_from_recon_p0005-p5_img_f${fr}g1d0b0.hv
#    done
#    stir_math sum_over_frames_test_${it} mult_${it}_f${fr}.hv
#    for fr in `count 24 28`; do
#    stir_math --including-first --accumulate sum_over_frames_test_${it}.hv mult_${it}_f${fr}.hv
#    done
#done
#
#stir_math -s sum_frame_sinograms fwd_dyn_from_p0005-p5_f23g1d0b0.hs
#for fr in `count 24 28`; do
#    stir_math -s --including-first --accumulate sum_frame_sinograms.hs fwd_dyn_from_p0005-p5_f${fr}g1d0b0.hs
#done
cd .. ;  
# rm -fr test_modelling_output

echo " " 
echo "The parametric imaging tests are OK. "
echo "You can remove all output with \"rm -fr test_modelling_output\""
