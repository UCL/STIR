#!/bin/csh -fe

if ($# < 3) then
    echo "usage: $0 jobname jobnum1 jobnum2 ..."
    echo "with jobnum? are integer >= 0"
    exit 1
endif

set jobname=$1

# skip jobname
shift

@ numjobs = $#


set jobdirpre = job_${jobname}_
set jobdir = ${jobdirpre}combined
echo "output will be in  $jobdir"
mkdir $jobdir

set max_iter=2000
set iter_inc=200
set do_randoms_estimate_stats=0
set is_iterative=1

cd $jobdir
set total_num_trials=0
foreach jobnum ($*)
   if ( -r   ../${jobdirpre}${jobnum}/current_trial ) then
	@ total_num_trials += `cat ../${jobdirpre}${jobnum}/current_trial`
   endif
end
echo $total_num_trials > current_trial

echo "Total number of trials is $total_num_trials"

if ($do_randoms_estimate_stats) then
 	set cumname = cum_${jobname}
	foreach pow (1 2 3 4)
		set all_files=""
		foreach jobnum ($*)
			if ( -r   ../${jobdirpre}${jobnum}/${cumname}_pow$pow.hs ) then
			    set all_files = "$all_files  ../${jobdirpre}${jobnum}/${cumname}_pow$pow.hs"
			endif
			end

        	stir_math -s ${cumname}_pow$pow.hs $all_files
	end
endif # do_randoms_estimate_stats

set iter=$iter_inc
while ($iter <= $max_iter)
	if ($is_iterative) then
               	echo iter $iter
		set cumname = cum_${jobname}_$iter
		set statsname = all_$iter.stats
	else
		set cumname = cum_${jobname}
		set statsname = all.stats
		set iter = $max_iter # make sure we get out
	endif
 	foreach pow (1 2 3 4)
		set all_files=""
		foreach jobnum ($*)
		   if ( -r   ../${jobdirpre}${jobnum}/${cumname}_pow$pow.hv ) then
			#set all_files = "$all_files  ../${jobdirpre}${jobnum}/${cumname}_pow$pow.hv"
		      if ( -r ${cumname}_pow$pow.hv ) then
			  set stir_math_option=--accumulate
		      else
			  set stir_math_option=""
		      endif
		      stir_math ${stir_math_option} \
			    ${cumname}_pow$pow.hv \
			    ../${jobdirpre}${jobnum}/${cumname}_pow$pow.hv
		   endif
		end

        	#stir_math ${cumname}_pow$pow.hv $all_files
	end
	# if the first jobdir has a .stats file (that is readable), then combine stats files
	if ( -r ../${jobdirpre}$1/$statsname ) then
	  rm -f $statsname
	  foreach jobnum ($*)
		cat  ../${jobdirpre}${jobnum}/${statsname} >> $statsname
	  end
	endif
        @ iter += $iter_inc
end




