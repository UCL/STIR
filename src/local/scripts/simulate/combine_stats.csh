#! /bin/csh -fe
# used -e option to exit as soon as a non-zero status is returned by any command



if ($# != 1 && !($# == 2 && "$1" == "-s")) then
    echo "usage: $0 [-s] jobname"
    echo "with jobname as set in multi_trial.csh"
    echo computes estimate for the mean, variance, and the variance on the 
    echo variance estimate from the following files:
    echo current_trial, 'cum_${jobname}_pow[1234]'
    echo output will be in stats subdirectory
    exit 1
endif


if ("$1" == "-s") then
    set name = $2
    set ext=hs
    set stir_math = "stir_math -s"
else
    set stir_math = "stir_math"
    set name = $1
    set ext=hv
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
$stir_math --accumulate --including-first --times-scalar $a variance_$name.$ext
#### stddev
$stir_math --including-first --power 0.5 stddev_$name variance_$name.$ext


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
$stir_math --times-scalar $a --accumulate --power 2 variance_on_variance_$name.$ext variance_$name.$ext

# res += -4 mean s3
set a = -4
$stir_math --mult tmp_$name  ${cumname}_pow3.$ext mean_${name}.$ext
$stir_math --times-scalar $a --accumulate variance_on_variance_$name.$ext tmp_$name.$ext

# res *= 1/((N-3)*(N-2)
set a = `awk '{ print 1/(($1 - 3)*($1 - 2)) }' $num_trials`
$stir_math --accumulate --times-scalar $a --including-first variance_on_variance_$name.$ext

#### stddev_on_variance
$stir_math --including-first --power 0.5 stddev_on_variance_$name variance_on_variance_$name.$ext

rm tmp_$name.*
rm pow2_mean_$name.*

