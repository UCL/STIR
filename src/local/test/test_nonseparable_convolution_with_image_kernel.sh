#! /bin/sh
# $Id$
# A script to test the nonseparable image filtering with convolution.
# Author: Charalampos Tsoumpas

# we do this but it only works in bash. all the rest will work in sh
set -e
trap "echo ERROR" ERR
WORKSPACE=`pwd`

INPUTDIR=$WORKSPACE/local/test/input/

OUTPUTDIR=test_postfilter_output
mkdir -p $OUTPUTDIR
pushd $OUTPUTDIR
if [ ! -r $WORKSPACE/samples/generate_image.par ]; then
echo "The sample file 'generate_image.par' is missing." 
exit 1;
fi
generate_image $WORKSPACE/samples/generate_image.par 
z_voxels_num=`cat $WORKSPACE/samples/generate_image.par | grep "Z output" | awk '{print ($8-1)*.5}'`
create_a_point delta image.hv ${z_voxels_num} 0 0 1 
postfilter separable_filtered_image image.hv ${INPUTDIR}/postfilter_separable_with_image_kernel.par 
postfilter separable_filter_kernel_response delta.hv ${INPUTDIR}/postfilter_separable_with_image_kernel.par 

postfilter identical_image image.hv ${INPUTDIR}/postfilter_nonseparable_with_delta.par
echo "Compare the filtered image with the original when filtered with delta function."
compare_image image.hv identical_image.hv
echo "First comparison is OK"
postfilter nonseparable_filtered_image image.hv ${INPUTDIR}/postfilter_nonseparable_with_image_kernel.par
echo "Compare the nonseparable filtered image with separable filtered image."
compare_image separable_filtered_image.hv nonseparable_filtered_image.hv
echo "Second comparison is OK"
echo " " 
popd
rm -fr $OUTPUTDIR 
echo " " 
echo "The Nonseparable Image Postfiltering Tests are OK. " 
