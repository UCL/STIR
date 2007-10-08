#! /bin/csh -fe
# used -e option to exit as soon as a non-zero status is returned by any command

if ($# != 1 && !($# == 2 && (("$1" == "--parametric") || ("$1" == "-s") ) ) ) then
    echo "usage: $0 [-s | --parametric] jobname"
    echo "with jobname as set in multi_trial.csh"
    echo computes estimate for the mean, variance, standard deviation 
    echo and the variance on the variance estimate from the following files:
    echo current_trial, 'cum_${jobname}_pow[1234]'
    echo output will be in stats subdirectory
    exit 1
endif

if ("$1" == "-s") then
    set name = $2
    set ext=hs
    set stir_math = "stir_math -s"
else
    if ("$1" == "--parametric") then
	set name = $2
	set ext=img
	set stir_math = "stir_math --parametric"
    else
	set stir_math = "stir_math"
	set name = $1
	set ext=hv
    endif
endif

mkdir -p stats
cd stats
set cumname=../cum_${name}
# file with number of trial
set num_trials = ../current_trial

#### mean
# complicated way to compute 1/niter
set a = `awk '{ print 1/$1 }' $num_trials`
$stir_math --including-first --times-scalar $a mean_${name} ${cumname}_pow1.$ext

# mean^2
$stir_math --including-first --power 2 pow2_mean_$name mean_${name}.$ext

#### variance
set niter = `cat $num_trials`
$stir_math --times-scalar -$niter variance_$name ${cumname}_pow2.$ext pow2_mean_$name.$ext
# res *= 1/(N-1)
set a = `awk '{ print 1/($1 - 1) }' $num_trials`
$stir_math --accumulate --including-first --times-scalar $a --min-threshold 0 variance_$name.$ext # The variance should be always positive.  

#### covariance
if ("$1" == "--parametric") then
    stir_math --including-first --times-scalar $a mean_mult_${name} ${cumname}_mult.hv
    mult_image_parameters -o mult_mean_${name} -i mean_${name}.$ext
    stir_subtract covariance_${name} mean_mult_${name}.hv mult_mean_${name}.hv
endif
#### stddev
$stir_math --including-first --power 0.5 --min-threshold 0 stddev_$name variance_$name.$ext


#### variance_on_variance
# res = s4 + 3 N mean^4
set a = `awk '{ print 3*$1  }' $num_trials`
$stir_math --times-scalar $a --power 4  variance_on_variance_$name.$ext ${cumname}_pow4.$ext mean_${name}.$ext


# res += (6N-6) variance*mean^2
set a = `awk '{ print 6*$1-6  }' $num_trials`
$stir_math --mult tmp_$name.$ext  variance_$name.$ext pow2_mean_${name}.$ext
$stir_math --times-scalar $a --accumulate variance_on_variance_$name.$ext tmp_$name.$ext

# res += -(N^2-3)/N variance^2
set a = `awk '{ print -($1*$1-3)/$1  }' $num_trials`
$stir_math --times-scalar $a --accumulate --power 2  variance_on_variance_$name.$ext variance_$name.$ext 

# res += -4 mean s3
set a = -4
$stir_math --mult tmp_$name  ${cumname}_pow3.$ext mean_${name}.$ext
$stir_math --times-scalar $a --accumulate  variance_on_variance_$name.$ext tmp_$name.$ext  
$stir_math --accumulate --including-first --min-threshold 0  variance_on_variance_$name.$ext # The variance should be always positive.

# res *= 1/((N-3)*(N-2)
set a = `awk '{ print 1/(($1 - 3)*($1 - 2)) }' $num_trials`
$stir_math --accumulate --times-scalar $a --including-first variance_on_variance_$name.$ext

#### stddev_on_variance
$stir_math --including-first --power 0.5  --min-threshold 0  stddev_on_variance_$name variance_on_variance_$name.$ext

rm tmp_$name.*
rm pow2_mean_$name.*
rm mult_mean_$name.*
rm mean_mult_$name.*


