#! /bin/sh
# $Id$
#PBS -k eo 
#PBS -l vmem=1990mb

# Script to run simset simulations
# Authors: Pablo Aguiar, Kris Thielemans

# Still relies on a few things such as
# - files have to be in 1 byte format
# - template_phg.rec has to have image sizes filled-in correctly
# - scanner is hardwired to HR+


####################### Script inputs ##################################

# These directories have to be changed for differents users
DIR_SIMSET=/data/home/${USER}/simset
#DIR_INPUT=/data/home/${USER}/sim_pet2/data
DIR_INPUT=`pwd`
DIR_OUTPUT=${DIR_INPUT}/${SIM_NAME}



########################  Script code ###################################

if [ $# -ne 0 -o -z "${SIM_NAME}" ]; then
    echo "usage: $0 "
    echo environment variable SIM_NAME has to be defined
    echo Output data will be in sub-directory with this name
    exit 1
fi

if [ $# -ne 0 -o -z "${PHOTONS}" ]; then
    echo "usage: $0 "
    echo environment variable PHOTONS has to be defined
    echo Contains the number of positrons to run
    exit 1
fi

if [ $# -ne 0 -o -z "${EMISS_DATA}" ]; then
    echo "usage: $0 "
    echo environment variable EMISS_DATA has to be defined
    echo Emission Data Base Filename
    exit 1
fi


if [ $# -ne 0 -o -z "${ATTEN_DATA}" ]; then
    echo "usage: $0 "
    echo environment variable ATTEN_DATA has to be defined
    echo Attenuation Data Base Filename
    exit 1
fi

# exit on error
set -e

mkdir -p ${DIR_OUTPUT}

cd ${DIR_INPUT}

# Copying into DIR_OUTPUT what SimSET needs to run
cp template* ${DIR_OUTPUT}
cp ${EMISS_DATA}.v ${DIR_OUTPUT}/act${SIM_NAME}.dat
cp ${ATTEN_DATA}.v ${DIR_OUTPUT}/att${SIM_NAME}.dat
cp ${EMISS_DATA}.hv ${DIR_OUTPUT}
cp ${ATTEN_DATA}.hv ${DIR_OUTPUT}
cp det.rec ${DIR_OUTPUT} 

cd ${DIR_OUTPUT}

# Building bin.rec and phg.rec from templates
sed -e s#SIMSET_DIRECTORY#${DIR_SIMSET}# \
    -e s#INPUT_DIRECTORY#${DIR_INPUT}# \
    -e s#OUTPUT_DIRECTORY#${DIR_OUTPUT}# \
    -e s#BIN.REC#bin.rec# \
    -e s#PHOTONS#${PHOTONS}# \
  < template_phg.rec > phg.rec


sed -e s#SIMSET_DIRECTORY#${DIR_SIMSET}# \
    -e s#INPUT_DIRECTORY#${DIR_INPUT}# \
    -e s#OUTPUT_DIRECTORY#${DIR_OUTPUT}# \
  < template_bin.rec > bin.rec


# Building index.dat to input in makeindexfile
echo phg.rec > index${SIM_NAME}.dat
echo y >> index${SIM_NAME}.dat
echo y >> index${SIM_NAME}.dat
echo 0 >> index${SIM_NAME}.dat
echo y >> index${SIM_NAME}.dat
echo act${SIM_NAME}.dat >> index${SIM_NAME}.dat
echo 0 >> index${SIM_NAME}.dat
echo 0 >> index${SIM_NAME}.dat
echo 1 >> index${SIM_NAME}.dat
echo n >> index${SIM_NAME}.dat
echo n >> index${SIM_NAME}.dat
echo y >> index${SIM_NAME}.dat
echo y >> index${SIM_NAME}.dat
echo 0 >> index${SIM_NAME}.dat
echo y >> index${SIM_NAME}.dat
echo att${SIM_NAME}.dat >> index${SIM_NAME}.dat
echo 0 >> index${SIM_NAME}.dat
echo 0 >> index${SIM_NAME}.dat
echo 1 >> index${SIM_NAME}.dat
echo n >> index${SIM_NAME}.dat
echo n >> index${SIM_NAME}.dat

$DIR_SIMSET/bin/makeindexfile < index${SIM_NAME}.dat >& ${DIR_OUTPUT}/makeindex.log

$DIR_SIMSET/bin/phg phg.rec > ${DIR_OUTPUT}/log

mv ${DIR_OUTPUT}/act${SIM_NAME}.dat ${DIR_OUTPUT}/${EMISS_DATA}.v
mv ${DIR_OUTPUT}/att${SIM_NAME}.dat ${DIR_OUTPUT}/${ATTEN_DATA}.v
rm -f ${DIR_OUTPUT}/rec.stat ${DIR_OUTPUT}/*.weight2 ${DIR_OUTPUT}/*.count ${DIR_OUTPUT}/index${SIM_NAME}.dat
# gzip ${DIR_OUTPUT}/*weight*
rm ${DIR_OUTPUT}/rec.act_indexes ${DIR_OUTPUT}/rec.activity_image ${DIR_OUTPUT}/rec.att_indexes ${DIR_OUTPUT}/rec.attenuation_image
rm ${DIR_OUTPUT}/template_* ${DIR_OUTPUT}/det.rec 

convert_SimSET_STIR_splitted.sh 0 > /dev/null
