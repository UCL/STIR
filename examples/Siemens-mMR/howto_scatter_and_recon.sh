#! /bin/sh -e
# Example script to reconstruct MMR data with randoms, norm and scatter.
# Currently supposes you have randoms estimated already.

# Author: Kris Thielemans

# directory with some standard .par files
: ${pardir:=~/devel/STIR/examples/Siemens-mMR}

: ${sino_input:=sinospan11_f1g1d0b0.hs}

# ECAT8 norm file 
# note: variable name is used in correct_projdata.par
: ${ECATNORM:=Norm_20130515125744.n.hdr.STIR}
export ECATNORM

# output (or input if it exists already) for normalisation sinogram
: ${norm_sino_prefix:=fullnormfactorsspan11}

if [ -r ${norm_sino_prefix}.hs ]; then
  echo "Re-using existing ${norm_sino_prefix}.hs"
else
  echo "Creating ${norm_sino_prefix}.hs"
  OUTPUT=${norm_sino_prefix} INPUT=${sino_input} correct_projdata ${pardir}/correct_projdata.par > ${norm_sino_prefix}.log 2>&1
fi

# input files etc
# (one level up as we will change directory)
: ${atnimg:=~/data/mmr/umap-bed.hv}
data3d=../${sino_input}
norm3d=../${norm_sino_prefix}.hs
: ${randoms3d:=../MLrandoms.hs}
acf3d=../acf.hs # will be created if it doesn't exist yet
# image 0 everywhere where activity is zero, 1 elsewhere
: ${mask_image:=mask_image.hv}

# all the rest should not need changing

# create mask image if it doesn't exist yet
if [ -r ${mask_image} ]; then
  echo Reusing mask image ${mask_image}
else
  FWHMx=20 FWHMz=20 postfilter ${mask_image} ${atnimg} ${pardir}/postfilter_Gaussian.par
  stir_math --accumulate --including-first  --max-threshold .001 ${mask_image}
  stir_math --accumulate --including-first  --add-scalar -.00099 ${mask_image}
  stir_math --accumulate --including-first  --min-threshold 0 ${mask_image}
  stir_math --accumulate --including-first  --times-scalar 100002 ${mask_image}
fi

# put all output in a subdir
mkdir -p output

# sum atn images
stir_math output/summed_atnimg.hv ${atnimg} ${bedcoilatnimg}

# copy image there such that we don't have to worry about names/location
stir_math output/mask_image.hv ${mask_image}
cd output
mask_image=mask_image.hv
atnimg=summed_atnimg.hv

cp ${pardir}/scatter.par .

scatter_template=${pardir}/scatter_template.hs
num_scat_iters=5

# compute 3D ACFs
if [ -r ${acf3d} ]; then
  echo Reusing existing ACF ${acf3d}
else
  calculate_attenuation_coefficients --PMRT  --ACF ${acf3d} ${atnimg} ${data3d} 
fi

# we will do the scatter estimation in 2D to save time, so run SSRB
data2d=data3d_ssrb.hs
acf2d=acf3d_ssrb.hs
norm2d=norm3d_ssrb.hs
randoms2d=randoms3d_ssrb.hs
# run SSRB
# TODO figure out num segments from data3d
# exact value doesn't matter too much (higher means less noise)
num_segs_to_combine=9
do_ssrb_norm=0
SSRB $data2d $data3d $num_segs_to_combine 1 $do_ssrb_norm
SSRB $acf2d $acf3d $num_segs_to_combine 1 1 # do_norm=1 for ACF
SSRB $randoms2d $randoms3d $num_segs_to_combine 1 $do_ssrb_norm
# we need to get norm2d=1/SSRB(1/norm3d))
stir_math -s --including-first --power -1 invnorm3d.hs ${norm3d}
SSRB invnorm2d.hs invnorm3d.hs  $num_segs_to_combine 1 $do_ssrb_norm
stir_math -s --including-first --power -1  ${norm2d} invnorm2d.hs

# create mask in sinogram space from image
mysinomask=mask2d.hs
if [ -r ${mysinomask} ]; then 
  echo "Re-using existing ${mysinomask}"
else
  forward_project fwd_mask.hs ${mask_image}  ${data2d}  ${pardir}/forward_projector_ray_tracing.par > fwd_mask.log 2>&1
  # add 1 to be able to use create_tail_mask_from_ACFs (which expects ACFs, so complains if the threshold is too low)
  stir_math -s --add-scalar 1 --including-first  --accumulate fwd_mask.hs
  create_tail_mask_from_ACFs  --safety-margin 2 --ACF-threshold 1.1 --ACF-filename fwd_mask.hs --output-filename ${mysinomask} > ${mysinomask}.log 2>&1
fi

# calculate scatter in 2D

# note: need OSEM because of gaps
do_FBP=0 NUM_SUBSETS=28 do_mask=0 max_scale_factor=2 sinomask=${mysinomask} endN=${num_scat_iters} do_average_at_2=1 background_proj_data_file=${randoms2d} estimate_scatter.sh ${atnimg} ${data2d}  ${scatter_template} ${acf2d} ${norm2d}


# now go to 3D

N=${num_scat_iters}
#unscaled=scatter_estimate_wrong_scale_$N
scaled2d=scatter_estimate_$N.hs
scatter3d=scatter_estimate_3d.hs
# use upsample_and_fit to upsample to 3d
# this is complicated as the 2d scatter estimate was divided by norm2d, so we need to undo this
# unfortunately, currently the values in the gaps in the scatter estimate are not quite zero (just very small)
# so we have to first make sure that they are zero before we do any of this, otherwise the values after normalisation will be garbage
# we do this by min-thresholding and then subtracting the threshold. as long as the threshold is tiny, this will be ok
stir_math -s --including-first --min-threshold 1e-9 tmp1.hs ${scaled2d}
stir_math -s --including-first --add-scalar -1e-9 tmp2.hs tmp1.hs
# ok, we can multiply with the norm
stir_math -s --mult normscatter2d.hs tmp2.hs ${norm2d}
# now we need to tell the upsampler that it needs to apply norm3d
cat <<EOF > norm3d.par
Bin Normalisation parameters:=
    type:= From ProjData
    Bin Normalisation From ProjData :=
      normalisation projdata filename:= ${norm3d}
    End Bin Normalisation From ProjData:=
END:=
EOF
# call upsampler
# currently, we do not fit anymore to avoid noise etc
# note: need to specify data-to-fit for geometry. actual contents will be ignored (also need to give weights at present, but they will be ignored as well)
# warning: if you want to let it fit the factors, you need to set the weights appropriately.
upsample_and_fit_single_scatter --min-scale-factor 1 --max-scale-factor 1 --remove-interleaving 0  --output-filename ${scatter3d} --data-to-fit ${data3d} --data-to-scale normscatter2d.hs --weights ${data3d} --norm norm3d.par >  upsample_and_fit_single_scatter_3d.log 2>&1

# end of scatter estimation

# do an image reconstruction to check

# construct additive_sinogram for OSMAPOSL
stir_math -s total_background_3d.hs ${scatter3d} ${randoms3d}
stir_math -s --mult mult_factors_3d.hs  ${acf3d} ${norm3d}
stir_math -s --mult additive_sino_3d.hs total_background_3d.hs  mult_factors_3d.hs

#stir_subtract -s background_corrected_data3d.hs ${data3d}  total_background_3d.hs
#stir_math -s --mult precorrected_data3d.hs background_corrected_data3d.hs mult_factors_3d.hs
#ZOOM=.4 INPUT=precorrected_data3d.hs OUTPUT=final_activity_image_3d FBP3DRP FBP3DRP.par

INPUT=${data3d} OUTPUT=final_activity_image_3d NORM=mult_factors_3d.hs ADDSINO=additive_sino_3d.hs SUBSETS=14 SUBITERS=42 SAVEITERS=14 SENS=subset_sens RECOMP_SENS=1 OSMAPOSL ${pardir}/OSMAPOSLbackground.par > final_activity_image_3d.log 2>&1
