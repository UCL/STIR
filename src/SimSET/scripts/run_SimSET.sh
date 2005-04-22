#! /bin/sh
#PBS -k eo 
#PBS -l vmem=1990mb

if [ $# -ne 0 -o -z "${SIM_NUM}" ]; then
    echo "usage: $0 "
    echo environment variable SIM_NUM has to be defined
    exit 1
fi

# exit on error
set -e



DIR_SIMSET=/data/home/ctsoumpas/simset
DIR_INPUT=/data/home/ctsoumpas/sim_pet/data
DIR_OUTPUT=/data/home/ctsoumpas/sim_pet/data/RESULTS${SIM_NUM}
mkdir -p ${DIR_OUTPUT}



cd ${DIR_INPUT}

sed -e s#SIMSET_DIRECTORY#${DIR_SIMSET}# \
    -e s#INPUT_DIRECTORY#${DIR_INPUT}# \
    -e s#OUTPUT_DIRECTORY#${DIR_OUTPUT}# \
    -e s#BIN.REC#bin${SIM_NUM}.rec# \
  < template_phg.rec > phg${SIM_NUM}.rec


sed -e s#SIMSET_DIRECTORY#${DIR_SIMSET}# \
    -e s#INPUT_DIRECTORY#${DIR_INPUT}# \
    -e s#OUTPUT_DIRECTORY#${DIR_OUTPUT}# \
  < template_bin.rec > bin${SIM_NUM}.rec

${DIR_INPUT}/../bats/batindex $DIR_SIMSET $SIM_NUM >& ${DIR_OUTPUT}/makeindex.log

$DIR_SIMSET/bin/phg phg${SIM_NUM}.rec > ${DIR_OUTPUT}/log

cd ${DIR_OUTPUT}
rm -f rec.stat *.weight2 *.count


# More things... 
gzip *weight*
rm rec.act_indexes rec.activity_image rec.att_indexes rec.attenuation_image
#rm ../phgmath*
#rm ../index.dat
mv ${DIR_INPUT}/bin${SIM_NUM} .
mv DIR_INPUT/phg${SIM_NUM} .



