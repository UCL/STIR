#! /bin/bash
# $Id$
# A script to check for any changes in the code which didn't cause any changes in the results.
# Author: Charalampos Tsoumpas

# we do this but it only works in bash. all the rest will work in sh
set -e
trap "echo ERROR" ERR
#Run Parametric Reconstruction
WORKSPACE=`pwd`

INPUTDIR=$WORKSPACE/recon_test/input/
NUMSUBS=4 # 16 subsets create a difference in the direct method of more than 8%!!!
ITER=40
SAVITER=40
MAXSEG=3
export INPUTDIR
export NUMSUBS
export ITER
export MAXSEG
export SAVITER

PATH=$WORKSPACE/$DEST/utilities:$WORKSPACE/$DEST/utilities/ecat:$WORKSPACE/scripts/:$WORKSPACE/$DEST/modelling_utilities:$WORKSPACE/$DEST/recon_test:$WORKSPACE/$DEST/iterative/POSMAPOSL:$WORKSPACE/$DEST/iterative/POSSPS:$PATH

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

tmpvar="" ;
for fr in `count 1 23 `; do
 tmpvar="$tmpvar fwd_dyn_from_p0005-p5_f23g1d0b0.hs"
done
for fr in `count 24 28 `; do
 tmpvar="$tmpvar fwd_dyn_from_p0005-p5_f${fr}g1d0b0.hs"
done


#tmpvar=`count --pre  "fwd_dyn_from_p0005-p5_f" --post "g1d0b0.hs" 1 28`
conv_to_ecat7 -s fwd_dyn_from_p0005-p5.S $tmpvar

#copy_frame_info.sh --frame-info-only  time.fdef fwd_dyn_from_p0005-p5.S
#copy_frame_info.sh --frame-info-only time.fdef dyn_from_p0005-p5.img
for direct in OSMAPOSL OSSPS ; do 
cp ${INPUTDIR}P${direct}.par .
echo "Test the direct P${direct} Patlak Plot reconstruction"
rm -f P${direct}.txt; P${direct} P${direct}.par > P${direct}.txt
echo "Multiply the parametri8c images with the model matrix to get the correspoding dynamic images."
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
