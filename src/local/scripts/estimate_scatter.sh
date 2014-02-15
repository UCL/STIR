#! /bin/bash
# (very complicated) script to estimate scatter for PET
# Authors: Kris Thielemans and Nikolaos Dikaios
#
# Copyright 2005-2009, Hammersmith Imanet Ltd

# Script for including single scatter correction to the final emission image

#estimate_single_scatter:
#needs a template-proj_data with sizes for scatter output
#important: 2D only , and not arc-corrected, 2*num_views = num_detectors_per_ring, no axial compression
# will create FBP2D.par and template_osem.par unless FBP2Dpar and OSEMpar environment variables are set

if [ -z "${startN}" ]; then startN=0; fi
if [ -z "${endN}" ]; then endN=50; fi
if [ -z "${attenuation_threshold}" ]; then attenuation_threshold=1.04; fi
if [ -z "${scale_factor_per_sinogram}" ]; then scale_factor_per_sinogram=1; fi
if [ -z "${do_threshold}" ]; then do_threshold=1; fi
if [ -z "${do_mask}" ]; then do_mask=1; fi
if [ -z "${mask}" ]; then mask=mask$suffix.hv; fi
if [ -z "${min_scale_factor}" ]; then min_scale_factor=.4; fi
if [ -z "${max_scale_factor}" ]; then max_scale_factor=100; fi
if [ -z "${back_off}" ]; then back_off=0; fi
#if [ -z "${max_mask_length}" ]; then max_mask_length=100000; fi
if [ -z "${DORANDOM}" ]; then DORANDOM=0; fi # TODO only in ST_scatter.par
if [ -z "${do_FBP}" ]; then do_FBP=1; fi
if [ -z "${postfilter_FWHM}" ]; then postfilter_FWHM=20; fi
if [ -z "${do_average_at_2}" ]; then do_average_at_2=0; fi
if [ -z "${multiples_FWHM}" ]; then multiples_FWHM=0; fi
# note if multiples_FWHM>0, set multiples_norm


if [ $# -ne 3 -a $# -ne 4 ]; then
    echo "usage: $0 \\" 1>&2
    echo "   atten_image emission_proj_data_file template_proj_data [ ACFfile ] " 1>&2
    echo "If ACFfile is not given I will use attenuation_coefficients'$suffix'.hs" 1>&2
    echo "WARNING: atten_image has to be in interfile format right now" 1>&2
# TODO this is because of restriction in get_image_dimensions.sh etc (used in zoom_att_image.sh)
    exit 1
fi

set -e # exit on error
trap "echo ERROR in estimate_scatter.sh" ERR

atten_image=$1
emission_proj_data_file=$2
template_proj_data_file=$3
if [ $# -lt 4 ]; then
attenuation_coefficients=attenuation_coefficients$suffix.hs
else
attenuation_coefficients=$4
fi
if [ $# -lt 5 ]; then
do_norm=0
else
norm_factors=$5
do_norm=1;
fi

if [ -r scatter.par ]; then
scatterpar=scatter.par
else
system_type=`get_system_type.csh $emission_proj_data_file`
scatterpar=${STIR_DATA_DIR}/templates/${system_type}_scatter.par
fi

echo Using $scatterpar

if [ ! -r $atten_image ]
then
    echo "Input file $atten_image not found - Aborting"
    exit 1
fi
if  [ ! -r $emission_proj_data_file ]
then
    echo "Input file $emission_proj_data_file not found - Aborting"
    exit 1
fi
if [ ! -r $template_proj_data_file ]
then
    echo "Input file $template_proj_data_file not found - Aborting"
    exit 1
fi

# define reconstruct(output_prefix,input[, ACFs [, FBP2D.par]])
function reconstruct()
{
  if [ $# != 2 -a $# != 3 -a $# != 4 ]; then
   echo "function reconstruct should be called with 2 to 4 parameters"
   echo arguments : "$*"
   exit 1
  fi
  if [ -z "${recon_zoom}" ]; then
     export recon_zoom=.3
  fi
  if [ $# != 4 ]; then
    FBP2Dpar=FBP2D_filt${postfilter_FWHM}.par
  else
    FBP2Dpar=$4
  fi
  if [ ! -r ${FBP2Dpar} ]; then
  # make ${FBP2Dpar}
  cat <<EOF  > ${FBP2Dpar}
fbp2dparameters :=
input file := \${INPUT}
output filename prefix := \${OUTPUT}
zoom := \${recon_zoom}
xy output image size (in pixels) := -1
alpha parameter for ramp filter := .5
cut-off for ramp filter (in cycles) := 0.3
post-filter type:=Separable Cartesian Metz
Separable Cartesian Metz Filter Parameters :=
x-dir filter FWHM (in mm):= ${postfilter_FWHM}
y-dir filter FWHM (in mm):= ${postfilter_FWHM}
z-dir filter FWHM (in mm):= ${postfilter_FWHM}
end Separable Cartesian Metz Filter Parameters :=
EOF
  using_PMRTBP=0
  #if [ "`get_system_type.csh $2`" = "RX" ]; then
  if false; then
  using_PMRTBP=1
  cat <<EOF  >> ${FBP2Dpar}
Back projector type:= matrix
back projector using matrix parameters:=
  Matrix type := Ray Tracing
  Ray tracing matrix parameters :=
   number of rays in tangential direction to trace for each bin :=5
  End Ray tracing matrix parameters :=
end back projector using matrix parameters:=
EOF
fi
  cat <<EOF  >> ${FBP2Dpar}
end := 
EOF
  fi # creation of ${FBP2Dpar}

  if [ $# -ge 3 ]; then
    # find name for atten_corr, getting rid of path info
    INPUT=atten_corr_seg0_${2##*/}
    # now replace extension with .hs
    INPUT=${INPUT%%.*}.hs
    stir_math -s --mult --max_segment_num_to_process 0 ${INPUT} $2 $3
  else
    INPUT=$2
  fi
  OUTPUT=${1}
  export INPUT OUTPUT
  FBP2D ${FBP2Dpar} >& FBP2D_${OUTPUT}.log
  if [ $# -ge 3 ]; then
    rm -f ${INPUT%%hs}*s
  fi
} # end of reconstruct

# define reconstructOSEM(output_prefix,input, ACFs, scatter_sino_before_AC, background_sino_before_AC, initial_image[, OSEMpar] )
function reconstructOSEM()
{
  if [ $# != 6 -a $# != 7 ]; then
   echo "function reconstructOSEM should be called with 5 to 6 parameters"
   echo arguments : "$*"
   exit 1
  fi
  if [ $# != 7 ]; then
    OSEMpar=template_osem_filt${postfilter_FWHM}.par
  else
    OSEMpar=$7
  fi

  if [ ! -r ${OSEMpar} ]; then
  # make ${OSEMpar}
  cat <<EOF  > ${OSEMpar}
OSMAPOSLParameters :=

output filename prefix := \${OUTPUT}

objective function type:= PoissonLogLikelihoodWithLinearModelForMeanAndProjData
PoissonLogLikelihoodWithLinearModelForMeanAndProjData Parameters:=

input file := \${INPUT}

projector pair type := Matrix
  Projector Pair Using Matrix Parameters :=
  Matrix type := Ray Tracing
  Ray tracing matrix parameters :=
   number of rays in tangential direction to trace for each bin := 10
  End Ray tracing matrix parameters :=
  End Projector Pair Using Matrix Parameters :=

  Bin Normalisation type := From ProjData
    Bin Normalisation From ProjData :=
	normalisation projdata filename:= \${MULTIPLICATIVE_SINOGRAM}
  End Bin Normalisation From ProjData:=

sensitivity filename:=  \${SENSITIVITY_NAME}
recompute sensitivity := \${RECOMPUTE_SENSITIVITY}

additive sinogram := \${SINOGRAM_TO_ADD_IN_DENOMINATOR}
zero end planes of segment 0:= 0

end PoissonLogLikelihoodWithLinearModelForMeanAndProjData Parameters:=

number of subsets:= \${NUM_SUBSETS}
start at subiteration number:=\${START_SUBITERATION}
number of subiterations:= \${NUM_SUBITERATIONS}
save estimates at subiteration intervals:= \${SAVE_SUBITERATION_NUM}
;use subset sensitivities := 1
initial estimate := \${INITIAL_IMAGE} 
enforce initial positivity condition:= 0

post-filter type:=Separable Cartesian Metz
Separable Cartesian Metz Filter Parameters :=
x-dir filter FWHM (in mm):= ${postfilter_FWHM}
y-dir filter FWHM (in mm):= ${postfilter_FWHM}
z-dir filter FWHM (in mm):= ${postfilter_FWHM}
end Separable Cartesian Metz Filter Parameters :=

END :=
EOF
  fi # end of make ${OSEMpar}

  final_output=$1
  export INPUT=$2
  export OUTPUT=OSEM_${final_output}
  export MULTIPLICATIVE_SINOGRAM=$3
  scatter_sino=$4
  background_sino=$5
  export INITIAL_IMAGE=$6
  
  export SENSITIVITY_NAME=sensitivity_${INPUT##*/}
  SENSITIVITY_NAME=${SENSITIVITY_NAME%.*}.hv
  if [ -r ${SENSITIVITY_NAME} ]; then
    export RECOMPUTE_SENSITIVITY=0
  else
    export RECOMPUTE_SENSITIVITY=1
  fi
  export NUM_SUBSETS=8
  export START_SUBITERATION=1
  export NUM_SUBITERATIONS=8
  export SAVE_SUBITERATION_NUM=8

  if [ -z ${background_sino} ]; then
    total_background_sino=${scatter_sino}
  else
    total_background_sino=total_background_with_${scatter_sino%%*/}
    stir_math -s ${total_background_sino} ${scatter_sino} ${background_sino}
  fi
  export SINOGRAM_TO_ADD_IN_DENOMINATOR=AC_of_${total_background_sino%%*/}

  stir_math -s --mult ${SINOGRAM_TO_ADD_IN_DENOMINATOR} ${MULTIPLICATIVE_SINOGRAM} ${total_background_sino}
  OSMAPOSL ${OSEMpar} >& ${OUTPUT}.log
  rm -f ${SINOGRAM_TO_ADD_IN_DENOMINATOR%.*}.*s
  stir_math $final_output ${OUTPUT}_${NUM_SUBITERATIONS}.hv
}

if [ ! -r $attenuation_coefficients  ]; then
echo "Computing attenuation correction factors "
calculate_attenuation_coefficients --PMRT --ACF \
    $attenuation_coefficients \
    $atten_image $emission_proj_data_file
fi

echo "zooming attenuation image to reduce number of scatter points"
if [ -z "${zoomed_attenuation_image}" ]; then 
    zoomed_attenuation_image=${atten_image##*/}; 
    zoomed_attenuation_image=zoomed_${zoomed_attenuation_image%.*};
fi
if [ ! -r ${zoomed_attenuation_image}.hv ]; then
  if [ -z "${zoom_xy}" ]; then zoom_xy=.14; fi
  if [ -z "${zoom_z}" ]; then zoom_z=.25; fi
  zoom_att_image.sh ${zoomed_attenuation_image} ${atten_image} ${zoom_xy} ${zoom_z}
fi

if [ -z "${suffix}" ]; then
  suffix=""
fi

if [ -z "${activity_image_prefix}" ]; then activity_image_prefix=activity_image${suffix}; fi
if [ -z "${unscaled_prefix}" ]; then unscaled_prefix="scatter_estimate_wrong_scale${suffix}"; fi
if [ -z "${scaled_prefix}" ]; then scaled_prefix=scatter_estimate${suffix}; fi
if [ -z "${scatter_corrected_data_prefix}" ]; then scatter_corrected_data_prefix=scatter_corrected_data${suffix}; fi

if [ -z ${background_proj_data_file} ]; then
  projdata_to_fit=${emission_proj_data_file}
else
  projdata_to_fit=meas_minus_background${suffix}.hs
  if [ ! -r ${projdata_to_fit} ]; then
    stir_subtract -s ${projdata_to_fit}  ${emission_proj_data_file} ${background_proj_data_file}
  fi
fi

if [ $startN = 0 ]; then
if [ ! -z "${initial_image}" ]; then
  if [ -r "${initial_image}" ]; then
    echo "Using initial image ${initial_image}"
    activity_image=${initial_image%.hv}
  else
    echo "ERROR: initial image ${initial_image} does not exist" 1>&2
    exit 1
  fi
else
  echo "reconstructing initial image"
  activity_image=${activity_image_prefix}_0
  reconstruct ${activity_image} ${projdata_to_fit} ${attenuation_coefficients} ${FBP2Dpar}
fi
else
  activity_image=${activity_image_prefix}_${startN}
fi

if [ $do_mask = 1 ]; then
  echo mask
  if [ ! -r ${mask} ]; then
   echo "CREATE MASK YOURSELF"
   exit 1
   FWHM=20 postfilter tmp${suffix}2$$.hv ${atten_image} $STIR_DATA_DIR/templates/postfilter_Gaussian3D.par 
   zoom_image --template ${activity_image}.hv tmp${suffix}$$ tmp${suffix}2$$.hv
   stir_math --including-first --min-threshold 0 --max-threshold .005 --divide-scalar .005 ${mask} tmp${suffix}$$.hv
  else
    # make sure it's of the same dimensions as the final image
    zoom_image --template ${activity_image}.hv zoomed_mask${suffix}.hv ${mask}
    echo "WARNING: hard-wired 11.11111111 for zoom"
    stir_math --accumulate --divide-scalar 11.111111111  --including-first zoomed_mask${suffix}.hv
    mask=zoomed_mask${suffix}.hv
  fi 
fi # end if mask

LIST=`count $startN $endN`

for N in $LIST
do
nextN=`expr $N + 1`

if [ $N = 2 -a ${do_average_at_2} = 1 ]; then
  activity_image=${activity_image_prefix}_0_and_1
  stir_math ${activity_image_prefix}_tmp.hv ${activity_image_prefix}_0.hv ${activity_image_prefix}_1.hv
  stir_math --including-first --divide-scalar 2 ${activity_image}.hv ${activity_image_prefix}_tmp.hv
fi


# name for output of estimate_scatter 
unscaled=${unscaled_prefix}_$N

ACTIVITY_IMAGE=${activity_image}.hv


if [ $do_mask = 1 ]; then
  stir_math --including-first --mult ${ACTIVITY_IMAGE%.hv}_mask$$ ${ACTIVITY_IMAGE%.hv}.hv ${mask}
  ACTIVITY_IMAGE=${ACTIVITY_IMAGE%.hv}_mask$$.hv
fi # end if mask

if [ ! -z ${image_filter} ]; then
   echo Filtering activity image
   out=${ACTIVITY_IMAGE%.hv}_filter$$.hv
   postfilter $out ${ACTIVITY_IMAGE} ${image_filter} >& ${out%.hv}.log
   ACTIVITY_IMAGE=$out
fi

if [ $do_threshold = 1 ]; then
    echo Thresholding activity image to positive values
    stir_math --min-threshold 0 --including-first ${ACTIVITY_IMAGE%.hv}_thresholded$$ ${ACTIVITY_IMAGE%.hv}.hv
    ACTIVITY_IMAGE=${ACTIVITY_IMAGE%.hv}_thresholded$$.hv
fi # end threshold

DENSITY_IMAGE=${zoomed_attenuation_image}.hv
LOW_DENSITY_IMAGE=${zoomed_attenuation_image}.hv
SCATTER_LEVEL=1
TEMPLATE=$template_proj_data_file
OUTPUT_PREFIX=$unscaled
export ACTIVITY_IMAGE DENSITY_IMAGE LOW_DENSITY_IMAGE SCATTER_LEVEL TEMPLATE OUTPUT_PREFIX
echo estimate_scatter ${ACTIVITY_IMAGE} ${zoomed_attenuation_image} $template_proj_data_file \
    $unscaled.hs $scatterpar  > estimate_scatter_${OUTPUT_PREFIX}.log
estimate_scatter $scatterpar >> estimate_scatter_${OUTPUT_PREFIX}.log 2>&1 


# output of upsample_and_fit_single_scatter
scaled=${scaled_prefix}_$N.hs

# make mask in sinogram space
sinomask=mask${suffix}_threshold${attenuation_threshold}_backoff${back_off}.hs
if [ ! -r $sinomask ]; then
 if [ ${attenuation_threshold} != 1000  ]; then
    create_tail_mask_from_ACFs  --safety-margin ${back_off} --ACF-threshold ${attenuation_threshold} --ACF-filename ${attenuation_coefficients} --output-filename ${sinomask} >& ${sinomask}.log
 else
    # first make mask complement
    create_tail_mask_from_ACFs --safety-margin ${max_mask_length}  --ACF-threshold 1.5 --ACF-filename ${attenuation_coefficients} --output-filename ${sinomask} >& ${sinomask}.log
    # now do 1-complement
   stir_math -s --accumulate --including-first --times-scalar -1 --add-scalar 1 ${sinomask} >>  ${sinomask}.log
 fi

fi

if [ $N = 0 ]; then
  actual_min_scale_factor=.5;
  actual_max_scale_factor=.5;
else
  actual_min_scale_factor=${min_scale_factor};
  actual_max_scale_factor=${max_scale_factor};
fi

if [ ${multiples_FWHM} = 0 ]; then
  scaled1=$scaled
  minSF=${actual_min_scale_factor}
  maxSF=${actual_max_scale_factor}
else
  scaled1=${scaled_prefix}_before_multiples_$N.hs
  minSF=1;maxSF=1;
fi


cmd="upsample_and_fit_single_scatter --min-scale-factor ${minSF} --max-scale-factor ${maxSF} --remove-interleaving 1 --half-filter-width 3  --output-filename ${scaled1} --data-to-fit ${projdata_to_fit} --data-to-scale ${unscaled}.hs --weights ${sinomask}" 
echo $cmd > upsample_and_fit_single_scatter_$unscaled.log
$cmd >> upsample_and_fit_single_scatter_$unscaled.log 2>&1 

if [ ! ${multiples_FWHM} = 0 ]; then
  multfiltered=${scaled_prefix}_filtered_$N.hs
  FWHM=${multiples_FWHM} postfilter_viewgrams.sh $multfiltered $scaled1 $STIR_DATA_DIR/templates/postfilter_Gaussian-x.par >& ${scaled_prefix}_filtered_$N.log
  mult=${scaled_prefix}_with_mult_$N.hs
  stir_math -s --times-scalar ${multiples_norm} $mult $scaled1 $multfiltered

  cmd="upsample_and_fit_single_scatter --min-scale-factor ${actual_min_scale_factor} --max-scale-factor ${actual_max_scale_factor} --remove-interleaving 0 --half-filter-width 3  --output-filename ${scaled} --data-to-fit ${projdata_to_fit} --data-to-scale ${mult} --weights ${sinomask}" 
  echo $cmd > upsample_and_fit_single_scatter_${mult%.hs}.log
  $cmd > upsample_and_fit_single_scatter_${mult%.hs}.log 2>&1 
fi
 

if [ ${do_FBP} = 1 ]; then
  # do precorrection
  #emission_proj_data - scatter_estimate(n) - background
  stir_subtract -s ${scatter_corrected_data_prefix}_$N.hs $emission_proj_data_file  $scaled ${background_proj_data_file}
  reconstruct ${activity_image_prefix}_${nextN} ${scatter_corrected_data_prefix}_$N.hs ${attenuation_coefficients} ${FBP2Dpar}

else
  reconstructOSEM ${activity_image_prefix}_${nextN} ${emission_proj_data_file} ${attenuation_coefficients} ${scaled} "${background_proj_data_file}" ${ACTIVITY_IMAGE} ${OSEMpar}

fi

# use this in next iteration
activity_image=${activity_image_prefix}_$nextN

done
