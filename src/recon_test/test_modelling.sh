#! /bin/bash
#
#  Copyright (C) 2009 - 2011, Hammersmith Imanet Ltd
#  Copyright (C) 2013 - 2014, University College London
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
# This script currently needs to be executed in the parent-directory of "recon_test"
# (This could be changed by adjusting INPUTDIR below).

# Author: Charalampos Tsoumpas
# Author: Kris Thielemans

# we do the next line such that the script aborts at any error (i.e. any non-zero return value of a command).
# Note that it only works in bash. all the rest will work in sh (except for a substition line, look for //)
set -e
trap "echo ERROR" ERR
#Run Parametric Reconstruction
WORKSPACE=`pwd`

INPUTDIR=$WORKSPACE/recon_test/input/
NUMSUBS=4 # 16 subsets create a difference in the direct method of more than 8%!!!
ITER=40
SAVITER=40
MAXSEG=1
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

command -v generate_image >/dev/null 2>&1 || { echo "generate_image not found or not executable. Aborting." >&2; exit 1; }
command -v get_dynamic_images_from_parametric_images >/dev/null 2>&1 || { echo "get_dynamic_images_from_parametric_images not found or not executable. Aborting." >&2; exit 1; }
command -v conv_to_ecat7 >/dev/null 2>&1 || { echo "conv_to_ecat7 not found or not executable. Aborting." >&2; exit 1; }

echo "Using executables like the following"
echo "Using `command -v get_dynamic_images_from_parametric_images`"
echo "Using `command -v fwdtest`"

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
conv_to_ecat7 p0005-p5.img p0005.hv p5.hv "ECAT 931"

# Create template images
if [ ! -r all1.img ]; then
generate_image ${INPUTDIR}generate_all1.par 
conv_to_ecat7 all1.img all1.hv all1.hv all1.hv all1.hv all1.hv all1.hv all1.hv all1.hv all1.hv all1.hv all1.hv all1.hv all1.hv all1.hv all1.hv all1.hv all1.hv all1.hv all1.hv all1.hv all1.hv all1.hv all1.hv all1.hv all1.hv all1.hv all1.hv all1.hv "ECAT 931" # 28 frames
fi

#copy_frame_info.sh --frame-info-only  time.fdef all1.img 

if [ ! -r all1-all1000.img ]; then
stir_math --including-first --times-scalar 1000 all1000 all1.hv
conv_to_ecat7 all1-all1000.img all1.hv all1000.hv "ECAT 931"
fi
cp all1.img dyn_from_p0005-p5.img
cp all1.img dyn_from_recon_p0005-p5.img
cp all1.img dyn_sens.img
cp all1-all1000.img indirect_Patlak.img

get_dynamic_images_from_parametric_images dyn_from_p0005-p5.img p0005-p5.img  ${INPUTDIR}PatlakPlot.par
#copy_frame_info.sh --frame-info-only  ${INPUTDIR}time.fdef dyn_from_p0005-p5.img
apply_patlak_to_images indirect_Patlak.img dyn_from_p0005-p5.img ${INPUTDIR}PatlakPlot.par

echo "Test the 'get_dynamic_images_from_parametric_images'"

# Create the appropriate files
ifheaders_for_ecat7 dyn_from_p0005-p5.img  < /dev/null
# if [ ! -r fwd_dyn_from_p0005-p5.S ]; then
rm -f fwd.par
cat <<EOF > fwd.par
Forward Projector parameters:=
type :=  ray tracing
  forward  Projector Using Ray Tracing Parameters :=
  End Forward Projector Using Ray Tracing Parameters :=
end :=
EOF

for fr in `count 23 28`; do
    fwdtest fwd_dyn_from_p0005-p5_f${fr}g1d0b0 ${INPUTDIR}ECAT_931_projdata_template.hs dyn_from_p0005-p5_img_f${fr}g1d0b0.hv fwd.par < /dev/null
done
#fi

# create "fake" normalisation file just for testing.
# we will use a sinogram full of ones (which won't influence the result)
stir_math -s --including-first --max_segment_num_to_process $MAXSEG --times-scalar 0 --add-scalar 1 all_ones.hs fwd_dyn_from_p0005-p5_f23g1d0b0.hs

tmpvar="" ;
for fr in `count 1 23 `; do
 tmpvar="$tmpvar fwd_dyn_from_p0005-p5_f23g1d0b0.hs"
done
for fr in `count 24 28 `; do
 tmpvar="$tmpvar fwd_dyn_from_p0005-p5_f${fr}g1d0b0.hs"
done

#tmpvar=`count --pre  "fwd_dyn_from_p0005-p5_f" --post "g1d0b0.hs" 1 28`

# conv_to_ecat7 -s fwd_dyn_from_p0005-p5.S $tmpvar
# make interfile dynamic data by concatenation and creating (by adjustment) a new header
cat ${tmpvar//.hs/.s} > fwd_dyn_from_p0005-p5.s
sed -e 's/number of time frames := 1/number of time frames := 28/' \
   -e 's/name of data file :=.*/name of data file := fwd_dyn_from_p0005-p5.s/' \
   fwd_dyn_from_p0005-p5_f23g1d0b0.hs > fwd_dyn_from_p0005-p5.hs

# TODO we need to fill in timing info really, but currently we'll pass time.fdef in the .par file instead
#copy_frame_info.sh --frame-info-only  time.fdef fwd_dyn_from_p0005-p5.S
#copy_frame_info.sh --frame-info-only time.fdef dyn_from_p0005-p5.img
for direct in OSMAPOSL OSSPS ; do 
cp ${INPUTDIR}P${direct}.par .
echo "Test the direct P${direct} Patlak Plot reconstruction"
rm -f P${direct}.txt; 
INPUT=fwd_dyn_from_p0005-p5.hs ${MPIRUN} P${direct} P${direct}.par > P${direct}.txt 2>&1
echo "Multiply the parametric images with the model matrix to get the corresponding dynamic images."
get_dynamic_images_from_parametric_images dyn_from_recon_p0005-p5.img P${direct}_${ITER}.img  ${INPUTDIR}PatlakPlot.par
get_dynamic_images_from_parametric_images dyn_sens.img sens.img ${INPUTDIR}PatlakPlot.par
ifheaders_for_ecat7 p0005-p5.img < /dev/null
ifheaders_for_ecat7 indirect_Patlak.img < /dev/null
ifheaders_for_ecat7 P${direct}_${ITER}.img  < /dev/null

ifheaders_for_ecat7 dyn_from_p0005-p5.img < /dev/null
ifheaders_for_ecat7 dyn_from_recon_p0005-p5.img < /dev/null
ifheaders_for_ecat7 dyn_sens.img  < /dev/null

echo "Compare the parametric images"
echo "indirect to original"
for par in 1 2; do
    compare_image -t .01 indirect_Patlak_img_f${par}g1d0b0.hv p0005-p5_img_f${par}g1d0b0.hv  
done
echo "direct to original"
for par in 1 2; do
    compare_image -t .01 P${direct}_${ITER}_img_f${par}g1d0b0.hv p0005-p5_img_f${par}g1d0b0.hv  
done
echo "direct to indirect"
for par in 1 2; do
    compare_image -t .01 P${direct}_${ITER}_img_f${par}g1d0b0.hv indirect_Patlak_img_f${par}g1d0b0.hv  
done
echo "Comparison is OK"

echo "Compare the dynamic images"
for fr in `count 23 28`
do 
    compare_image -t .01 dyn_from_recon_p0005-p5_img_f${fr}g1d0b0.hv dyn_from_p0005-p5_img_f${fr}g1d0b0.hv 
done
echo "Comparison is OK"

echo "Test the utility: 'mult_model_with_dyn_images'"
echo "Multiply the  dynamic images with the model matrix to get images in the parametric space."
#if [ ! -r test_mult_dyn_with_model.img ]; then
    cp indirect_Patlak.img test_mult_dyn_with_model.img
#fi
mult_model_with_dyn_images test_mult_dyn_with_model.img dyn_from_p0005-p5.img ${INPUTDIR}PatlakPlot.par
ifheaders_for_ecat7 test_mult_dyn_with_model.img < /dev/null

rm -f manip_image_counts.inp ; echo 9 > manip_image_counts.inp ; echo 0 >> manip_image_counts.inp
echo " "
echo "Min Counts for Par 1 "

value_1=`manip_image test_mult_dyn_with_model_img_f1g1d0b0.hv <manip_image_counts.inp 2>&1 |  grep "Max" |awk '{print $6}' `
is_differ=`echo ${value_1} | awk ' { print ($1>.0001) } '` 
if [  ${is_differ} -eq 1 ]; then   
    echo "When estimate min_counts_in_images values do not match. Check mult_model_with_dyn_images.cxx"
    exit 1
else
    echo " Test OK "
fi

echo "Min Counts for Par 2 "
value_2=`manip_image test_mult_dyn_with_model_img_f2g1d0b0.hv <manip_image_counts.inp 2>&1 |  grep "Max" |awk '{print $6}' `
is_differ=`echo ${value_2} | awk ' { print ($1>.0001) } '` 
if [  ${is_differ} -eq 1 ]; then   
    echo "When estimate min_counts_in_images values do not match. Check mult_model_with_dyn_images.cxx"
    exit 1
else 
    echo " Test OK "
fi

echo "Max Counts for Par 1 "
value_3=`manip_image test_mult_dyn_with_model_img_f1g1d0b0.hv <manip_image_counts.inp 2>&1 |  grep "Max" |awk '{print $7}' `
# 6.391818 is the scale due to the zoom factor.
is_differ=`echo ${value_3} | awk ' { print (($1-1619.32*6.391818*6.391818)*($1-1619.32*6.391818*6.391818)>.5*6.391818*6.391818*6.391818*6.391818*.5) } '` 
if [  ${is_differ} -eq 1 ]; then   
    echo "When estimate max_counts_in_images values do not match. Check mult_model_with_dyn_images.cxx"
    exit 1
else
 echo " Test OK "
fi

echo "Max Counts for Par 2 "
value_4=`manip_image test_mult_dyn_with_model_img_f2g1d0b0.hv <manip_image_counts.inp 2>&1 |  grep "Max" |awk '{print $7}' `
is_differ=`echo ${value_4} | awk ' { print (($1-0.356552*6.391818*6.391818)*($1-0.356552*6.391818*6.391818)>.001*6.391818*6.391818*.001) } '` 
if [  ${is_differ} -eq 1 ]; then   
    echo "When estimate max_counts_in_images values do not match. Check mult_model_with_dyn_images.cxx"
    exit 1
else 
    echo " Test OK "
fi

# Extract Sensitivity Frames
ifheaders_for_ecat7 sens.img  < /dev/null

echo "The sensitivity image is not tested, yet!!!"
# Multiply the parametric images with the sensitivity images ### NOT IMPLEMENTED YET
for it in ${ITER}; do
    for fr in `count 23 28` ; do
	stir_math --mult mult_${it}_f${fr} dyn_sens_img_f${fr}g1d0b0.hv dyn_from_recon_p0005-p5_img_f${fr}g1d0b0.hv
    done
    stir_math sum_over_frames_test_${it} mult_${it}_f${fr}.hv
    for fr in `count 24 28`; do
    stir_math --including-first --accumulate sum_over_frames_test_${it}.hv mult_${it}_f${fr}.hv
    done
done

stir_math -s sum_frame_sinograms fwd_dyn_from_p0005-p5_f23g1d0b0.hs
for fr in `count 24 28`; do
    stir_math -s --including-first --accumulate sum_frame_sinograms.hs fwd_dyn_from_p0005-p5_f${fr}g1d0b0.hs
done
done # POSMAPOSL POSSPS#
cd .. ;  
# rm -fr test_modelling_output

echo " " 
echo "The Direct Reconstructions tests are OK. "
echo "You can remove all output with \"rm -fr test_modelling_output\""
