#! /bin/sh
# $Id$
#PBS -k eo 
# better to add this in the qsub statement PBS -l vmem=1900mb

# Script to run simset simulations
# Authors: Pablo Aguiar, Kris Thielemans

# Still relies on a few things such as
# - template_phg.rec has to have image sizes filled-in correctly
# - scanner is hardwired to HR+


####################### Script inputs ##################################

# These directories have to be changed for differents users
#SIMSET_DIR=/data/home/${USER}/simset



########################  Script code ###################################


if [ $# -ne 0 -o -z "${DIR_INPUT}" ]; then
    echo Environment variable DIR_INPUT will default to current directory.
    DIR_INPUT=`pwd`
fi

if [ $# -ne 0 -o -z "${DIR_OUTPUT}" ]; then
    if [ $# -ne 0 -o -z "${SIM_NAME}" ]; then
      echo "usage: $0 "
      echo environment variable SIM_NAME or DIR_OUTPUT has to be defined
      exit 1
    fi
    DIR_OUTPUT=${DIR_INPUT}/${SIM_NAME}
fi
echo Output data will be in ${DIR_OUTPUT}


if [ $# -ne 0 -o -z "${PHOTONS}" ]; then
    echo "usage: $0 "
    echo environment variable PHOTONS has to be defined
    echo Contains the number of positrons to run
    exit 1
fi

if [ $# -ne 0 -o -z "${EMISS_DATA}" ]; then
    echo "usage: $0 "
    echo environment variable EMISS_DATA has to be defined
    echo Emission Data Filename
    exit 1
fi


if [ $# -ne 0 -o -z "${ATTEN_DATA}" ]; then
    echo "usage: $0 "
    echo environment variable ATTEN_DATA has to be defined
    echo "Attenuation Data Filename (in mu-values units cm^-1)"
    exit 1
fi

num_seg=0
if [ ! -z "${NUM_SEG}" ]; then
num_seg=${NUM_SEG}
fi

convert_att_to_simset=1
if [ ! -z "${CONVERT_ATT_TO_SIMSET}" ]; then
convert_att_to_simset=${CONVERT_ATT_TO_SIMSET}
fi

if [ $# -ne 0 -o -z "${TEMPLATE_PHG}" ]; then
    TEMPLATE_PHG=template_phg.rec
fi
echo "Using ${TEMPLATE_PHG}"

if [ $# -ne 0 -o -z "${TEMPLATE_BIN}" ]; then
    TEMPLATE_BIN=template_bin.rec
fi
echo "Using ${TEMPLATE_BIN}"

# exit on error
set -e

mkdir -p ${DIR_OUTPUT}

cd ${DIR_INPUT}

# Copying into DIR_OUTPUT what SimSET needs to run
if [ ${DIR_INPUT} != ${DIR_OUTPUT} ]; then
  cp template* ${DIR_OUTPUT}
  cp det.rec ${DIR_OUTPUT} 
fi

cd ${DIR_OUTPUT}
# first convert input emission to 1byte data
cat > output_format_1byte.par  <<EOF
output file format parameters:=
output file format type:=interfile
Interfile Output File Format Parameters:=
number format:=signed integer
number_of_bytes_per_pixel:=1
end Interfile Output File Format Parameters:=
end :=
EOF
stir_math  --output-format output_format_1byte.par \
   act.dat ${DIR_INPUT}/${EMISS_DATA}
rm -f output_format_1byte.par

if [ $convert_att_to_simset == 1 ]; then
  # convert attenuation image to simset indices
  conv_to_SimSET_att_image att.dat ${DIR_INPUT}/${ATTEN_DATA}
else
  cp ${DIR_INPUT}/${ATTEN_DATA} att.dat
fi
# Building bin.rec and phg.rec from templates
sed -e s#SIMSET_DIRECTORY#${SIMSET_DIR}# \
    -e s#INPUT_DIRECTORY#${DIR_INPUT}# \
    -e s#OUTPUT_DIRECTORY#${DIR_OUTPUT}# \
    -e s#BIN.REC#bin.rec# \
    -e s#PHOTONS#${PHOTONS}# \
  < ${TEMPLATE_PHG} > phg.rec


sed -e s#SIMSET_DIRECTORY#${SIMSET_DIR}# \
    -e s#INPUT_DIRECTORY#${DIR_INPUT}# \
    -e s#OUTPUT_DIRECTORY#${DIR_OUTPUT}# \
  < ${TEMPLATE_BIN} > bin.rec


# Building index.dat to input in makeindexfile
echo phg.rec > index.dat
echo y >> index.dat
echo y >> index.dat
echo 0 >> index.dat
echo y >> index.dat
echo act.dat >> index.dat
echo 0 >> index.dat
echo 0 >> index.dat
echo 1 >> index.dat
echo n >> index.dat
echo n >> index.dat
echo y >> index.dat
echo y >> index.dat
echo 0 >> index.dat
echo y >> index.dat
echo att.dat >> index.dat
echo 0 >> index.dat
echo 0 >> index.dat
echo 1 >> index.dat
echo n >> index.dat
echo n >> index.dat

$SIMSET_DIR/bin/makeindexfile < index.dat >& ${DIR_OUTPUT}/makeindex.log
$SIMSET_DIR/bin/phg phg.rec >& ${DIR_OUTPUT}/log

rm -f ${DIR_OUTPUT}/rec.stat ${DIR_OUTPUT}/*.weight2 ${DIR_OUTPUT}/*.count ${DIR_OUTPUT}/index.dat
# gzip ${DIR_OUTPUT}/*weight*
#rm ${DIR_OUTPUT}/rec.act_indexes ${DIR_OUTPUT}/rec.activity_image ${DIR_OUTPUT}/rec.att_indexes ${DIR_OUTPUT}/rec.attenuation_image
if [ ${DIR_INPUT} != ${DIR_OUTPUT} ]; then
  rm ${DIR_OUTPUT}/template_*
fi

convert_SimSET_STIR_splitted.sh ${num_seg} >& log_convert_SimSET_STIR
