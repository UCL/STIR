#! /bin/sh
if [ $# -lt 2 ]; then
    echo "usage: $0 [--max-iter n ] jobname jobnum1 [jobnum2 ...]"
    exit 1
fi
max_iter=0
if [ "$1" = --max-iter ]; then
  max_iter=$2
  shift
  shift
fi 
# exit on error
set -e
iter_inc=200

jobname=$1
shift

if [ $max_iter -gt 0 ]; then
    is_iterative=1
else
    is_iterative=0
    max_iter=$iter_inc
fi

curdir=`pwd`
while [ $# -ge 1 ]; do
    echo Processing ${jobname}_$1
    cd "job_${jobname}_$1"
    for iter in `count $iter_inc $max_iter $iter_inc`; do
	if [ $is_iterative -ne 0 ]; then
		name=_$iter.stats
		pattern="${jobname}*_${iter}_*.stats"
	else
		name=.stats
		pattern="${jobname}*_*.stats"
	fi
	echo Making all$name
	rm -f all$name
	for s in $pattern; do tail -n +3 $s >> all$name; done

    done
    cd "$curdir"
    shift
done
echo Done