#!/bin/csh -fe
# use -e such that script exits immediately when a non-zero status is returned

if ($#argv != 1) then
    echo "usage: $0 jobnum"
    echo "with jobnum an integer >= 0"
    echo "the jobnum is used to initalise the seed such that there's no overlap"
    echo "in trials between different jobs"
    exit 1
endif

# uses:
# ${file}.par, assumes that output is on image ${file}*.hv 
# precorrect_$dataname.par #and ${dataname}_ROI.par
# $mean_of_prompts.hs, $mean_randoms.hs

set dataname=fesslers_phantom_with_hotcoldzero
set do_ROI=1
#set sourcedir=`pwd`/ # use trailing backslash
set sourcedir=~/smallsim/larger/
cd $sourcedir
set workdir=$sourcedir

set do_randoms_estimate=0

    set randoms_estimation_method=meanrandoms
    set do_randoms_estimate_stats=0
    #name for output file with randoms estimate, as expected by .par files
    set randoms_estimate=randoms_estimate
    #name of input file to generate 'delayed'
    #note: ML randoms estimate will use ${mean_randoms}_span1
    set mean_randoms=mean_randoms
    # used to SSRB the noisy randoms to required dimensions
    set span=3
    set mash_factor=1

set jobname=EM_hotcoldzero_less_noise_$randoms_estimation_method

set mean_of_prompts="mean_prompts_with_hotcoldzero"
#name of 'measured' prompts as expected by .par files
set prompts="prompts_noise"

set do_precorrection=0 
set do_prepare_data=0
# scatter term to be subtracted AFTER norm+atten correction by precorrect_$dataname.par
# or added to the 'additive sinogram' term
# leave empty if not necessary

set reconprog=OSMAPOSL  # set to empty if no reconstruction required
set max_iter=2000 # set to 0 if non-iterative reconstruction
set iter_inc=200


set parfile=EM_prompts_noise_$max_iter
set outputfile=EM_prompts_noise


set num_trials=50
set seed_multiplication_factor=100
set noise_factor=1


# end of configuration

set jobnum = $1
set jobdir = ${workdir}job_${jobname}_${jobnum}
echo "output will be in  $jobdir"
mkdir -p $jobdir
cd $jobdir

set progpath="nice +18 "

set ROIparfile=$sourcedir"${dataname}_ROI.par"
if ("%%$reconprog" == "%%") then
  set do_reconstruction=0
else
  set do_reconstruction=1
endif
if ($max_iter == 0) then
  set is_iterative=0
  set iter_inc=1
  set max_iter=$iter_inc 
    # KT necessary for loop for cum
else
  set is_iterative=1
endif


if ($do_reconstruction) then
# you might have to adjust these
ln -s ${sourcedir}attenuation_correction_factors.hs
ln -s ${sourcedir}attenuation_correction_factors.s
#ln -s ${sourcedir}scatter.hs
#ln -s ${sourcedir}scatter.s
ln -s ${sourcedir}sens.hv
ln -s ${sourcedir}sens.v
ln -s ${sourcedir}mean_prompts_denominator.hs
ln -s ${sourcedir}mean_prompts_denominator.s
endif

# do not modify anything else

set trial = 1
@ seed = $seed_multiplication_factor * $num_trials * $jobnum

# important: poisson_noise should be called with -p to preserve the mean
# of the counts (independent of the noise_factor)
# otherwise, the randoms (and scatter) would have to be scaled with noise_factor as well
while ($trial <= $num_trials )
        echo trial $trial
	echo $trial > current_trial_start
	if ( -r $randoms_estimate.hs ) \
	  rm -f $randoms_estimate.*s
	if ($do_randoms_estimate) then
		# do this only if we estimate randoms background from delayed
		@ seed ++
		# add any randoms estimation here
		if ($randoms_estimation_method == delayed) then
			${progpath}poisson_noise -p $randoms_estimate ${sourcedir}${mean_randoms}.hs $noise_factor $seed >&/dev/null

		else
			# ML estimation
			${progpath}poisson_noise -p delayed_randoms_span1 ${sourcedir}${mean_randoms}_span1.hs $noise_factor $seed >&/dev/null

			${progpath}find_ML_singles_from_delayed estimated_singles delayed_randoms_span1.hs 20 </dev/null
			${progpath}apply_normfactors3D ML_randoms_estimate.s estimated_singles ${sourcedir}smallspan1.hs 1 1 20 1 0 0
			set randoms_estimate_span1=ML_randoms_estimate
			# now span the randoms estimate
			SSRB $randoms_estimate.s $randoms_estimate_span1.hs $span $mash_factor 0
			# TODO check if mean_value of above estimate is equal to the mean (in scale anyway)
		endif
		# keep cumulative sums of randoms_estimate for mean and variance

		if ($do_randoms_estimate_stats) then
			set cumname=cum_${jobname}
			if ($trial == 1) then
                        	${progpath}ifcphs $randoms_estimate ${cumname}_pow1
				foreach pow (2 3 4)
        	                  ${progpath}stir_math -s --power $pow --including-first ${cumname}_pow$pow.hs $randoms_estimate.hs
				end
	                else
				foreach pow (1 2 3 4)
                	          ${progpath}stir_math -s --power $pow  --accumulate ${cumname}_pow$pow.hs $randoms_estimate.hs
				end
	                endif
	        endif # do_randoms_estimate_stats
	else
	  # no randoms_estimate
 	  # use mean_randoms as randoms_estimate
	  ln -s   ${sourcedir}$mean_randoms.hs $randoms_estimate.hs
	endif # do_randoms_estimate

        if ($do_reconstruction) then

	    # generate prompts with added noise
	    @ seed ++ 
	    ${progpath}poisson_noise -p $prompts ${sourcedir}$mean_of_prompts.hs $noise_factor $seed
	    if ($do_precorrection) then
	        # subtract randoms from prompts and do other corrections
	        ${progpath}correct_projdata ${sourcedir}precorrect_$dataname.par
	        if ($reconprog == FBP3DRP || $reconprog == FBP2D) then
		  # pretend arccorrection for FBP3DRP
		  sed -e "s/applied corrections := {None}/applied corrections := {arc correction}/" \
			< precorrected.hs >precorrected_arccorrected.hs
            
		endif
	    endif

	    if ($do_prepare_data) then
		prepare_data ${sourcedir}prepare_$dataname.par
		#if ($do_randoms_estimate) then
		#	stir_math -s --mult randoms_estimate_times_normatten.hs \
		#		$randoms_estimate.hs atten_nrm_factors.hs
		#endif
	    endif

	    # reconstruct

	    echo "${progpath}${reconprog} ${sourcedir}${parfile}.par >& ${parfile}.log"
	    ${progpath}${reconprog} ${sourcedir}${parfile}.par >& /dev/null
#${parfile}.log

	    set iter=$iter_inc
	    while ($iter <= $max_iter)
		if ($is_iterative) then
                	echo iter $iter
	                set outfile=${jobname}_${iter}_$trial.stats
        	        set imagefile=${outputfile}_$iter.hv
			set cumname=cum_${jobname}_$iter
		else
	                set outfile=${jobname}_$trial.stats
        	        set imagefile=${outputfile}.hv
			set cumname=cum_${jobname}
		endif
		# calc ROI statistics
		if ($do_ROI) then
                  echo "${progpath}list_ROI_values $outfile $imagefile $ROIparfile"
                  ${progpath}list_ROI_values $outfile $imagefile $ROIparfile
                endif

		# save cumulative sums
		if ($trial == 1) then
                        ${progpath}ifcp $imagefile ${cumname}_pow1
			foreach pow (2 3 4)
                          ${progpath}stir_math  --power $pow --including-first ${cumname}_pow${pow}.v $imagefile
			end
                else
			foreach pow (1 2 3 4)
                          ${progpath}stir_math  --power $pow --accumulate  ${cumname}_pow${pow}.hv $imagefile
			end
                endif
                @ iter += $iter_inc
	    end
	endif # do_reconstruction
	echo $trial > current_trial
        @ trial ++
end




