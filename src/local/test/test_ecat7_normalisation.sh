#! /bin/bash 
# $Id$
# This is a script that compares results from STIR's correct_projdata
# when using ECAT7 norm files and CTI's bkproj_3D. Results should obviously
# identical.
# Author: Kris Thielemans

recompute_bkproj_3D=0
recompute_STIR=1
do_sgl_test=0
bkproj_3D="bkproj_3D_lln -N0"
CORRECT_PROJDATA=correct_projdata

function write_help() {
cat <<EOF
TO RUN DO THE FOLLOWING:

1. get an ECAT7 .S and .n file from somewhere, let's call them file.S and normfile.n
2. run
  ./test_normalisations.sh  file normfile.n
  Note: on Solaris, you'll probably have to do instead
   bash test_normalisations.sh  file normfile.n
3. if you have the corresponding .sgl file, copy it as well, and do
  ./test_normalisations.sh --do_sgl_test file normfile.n

run ./test_normalisations.sh for more extensive usage.
For example, if the normalised files already exists, the script will by default not recompute them.



Here is what it does:
- It uses bkproj3D to normalise file.S
- It uses correct_projdata to normalise file.S (where singles are read from the .S file)
- compare the sinograms. They should be almost identical.

If --do_sgl_test is used, there is an additional test:
- It uses correct_projdata to normalise file.S (where singles are read from the .sgl file)
- compare the sinograms



IF THERE IS A PROBLEM

You could always check if it is due to the dead-time correction or not.
You can switch off dead-time correction by setting all singles in the ECAT7 subheaders to 0.

Potentially also useful is to use ECAT7 data where the measured data is all set to 1. That way,
the normalised data will be just the normalisation factors.
You might be able to do this use a (local) STIR utility:

1. Copy file.S into ones.S
2. ifheaders_for_ecat7 ones.S
3. fillwith1 ones_f1g1d0b0.hs
EOF
} # end of function write_help


if [ $# -gt 0 ]; then
 while [ "${1:0:1}" = "-"  ]; do  # while 1st arg starts with -
   case "$1" in 
     --do_sgl_test) do_sgl_test=1; sgl_filename=$2; shift;shift;;
     --recompute_bkproj_3D) recompute_bkproj_3D=1; shift;;
     --do-not-recompute_STIR) recompute_STIR=0; shift;;
     --bkproj_3D) bkproj_3D=$2; shift;shift;;
     --correct_projdata) CORRECT_PROJDATA=$2; shift;shift;;
     --help) write_help; exit 1;;
     *) echo "Unrecognised option"; exit 1;;
   esac
done
fi

if [ $# != 2 ]; then
    echo "Usage: $0 \\"
    echo "   [--help] \\"
    echo "   [--do_sgl_test sgl_filename ] [--recompute_bkproj_3D] [--do-not-recompute_STIR] \\"
    echo "   [--bkproj_3D  bkproj_3D_filename] [--correct_projdata correct_projdata_filename] \\"
    echo "   ecat7_filename_without.S normfile"
    echo ""
    echo "bkproj_3D_filename defaults to bkproj_3D_lln"
    echo "correct_projdata_filename defaults to correct_projdata"
    exit 1
fi

set -e #exit on error
# note: next line doesn't work on Solaris sh. You could remove it, or use bash instead
trap "echo ERROR executing $0. Check '*'.log files" ERR

scanname=$1
normfile=$2

rm -f ifheaders_for_ecat7.log
ifheaders_for_ecat7 ${scanname}.S</dev/null 2> ifheaders_for_ecat7.log

numframes=`get_time_frame_info --num-time-frames  $scanname.S`
allframes=`count 1 $numframes`
rm -f ifheaders_for_ecat7.log

bkproj3d_prefix=normalised_bkproj3d_${scanname}
stir_prefix=normalised_stir_${scanname}
stir_sgl_prefix=normalised_stir_with_sgl_${scanname}

for frame_num in $allframes; do
    acquisition_spec=f${frame_num}g1d0b0
    if [ $recompute_bkproj_3D = 1 -o ! -r ${bkproj3d_prefix}_${acquisition_spec}.S ]; then
      echo Running $bkproj_3D to normalise  ${bkproj3d_prefix}_${acquisition_spec}.S
      # note -c: switch off arc-correction, -Q do not reconstruct
	echo ${bkproj_3D} -s ${scanname}.S  -D15 -c -Q -M ${bkproj3d_prefix}_${acquisition_spec}.S -n ${normfile} -m${frame_num},1,1,0,0 
	${bkproj_3D} -s ${scanname}.S  -D15 -c -Q -M ${bkproj3d_prefix}_${acquisition_spec}.S -n ${normfile} -m${frame_num},1,1,0,0 >& ${bkproj3d_prefix}_${acquisition_spec}.log   
      ifheaders_for_ecat7 ${bkproj3d_prefix}_${acquisition_spec}.S </dev/null 2>> ifheaders_for_ecat7.log
    else
      echo Using existing file ${bkproj3d_prefix}_${acquisition_spec}.S
    fi
    if [  $recompute_STIR = 1 -o ! -r ${stir_prefix}_${acquisition_spec}.hs ]; then
	## make a par file for correct_projdata with ecat7
        parfile=correct_with_ecat7_${scanname}_${acquisition_spec}.par
        rm -f ${parfile}
	cat  <<EOF > ${parfile}
	correct_projdata Parameters := 
	    input file :=  ${scanname}_S_${acquisition_spec}.hs
	    output filename := ${stir_prefix}_${acquisition_spec}
	    Bin Normalisation type:= From ECAT7
	    Bin Normalisation From ECAT7:=
		singles rates := Singles From ECAT7
		Singles Rates From ECAT7 :=
		    ECAT7_filename :=${scanname}.S
		End Singles Rates From ECAT7:=
		normalisation_filename:= ${normfile}
	    End Bin Normalisation From ECAT7:=
	    time frame definition filename := ${scanname}.S
	    time frame number:= ${frame_num} 
	END:= 
EOF
	echo Executing ${CORRECT_PROJDATA} ${parfile}
	${CORRECT_PROJDATA} ${parfile}
    else
        echo Using existing file ${stir_prefix}_${acquisition_spec}.hs
    fi

    #now do the correction on projdata where singles are taken from sgl file
    # this will only be tested if singles need to be taken into account ( singles set to 1 in argument list) 
    if [ ${do_sgl_test} = 1 ]; then
        if [ $recompute_STIR = 1 -o ! -r ${stir_sgl_prefix}_${acquisition_spec}.hs ]; then
	# make .par file
        parfile=correct_with_ecat7_sgl_${scanname}_${acquisition_spec}.par
        rm -f ${parfile}
	cat > ${parfile} <<EOF 
	correct_projdata Parameters := 
	    input file :=  ${scanname}_S_${acquisition_spec}.hs
	    output filename := ${stir_sgl_prefix}_${acquisition_spec}
	    Bin Normalisation type:= From ECAT7
	    Bin Normalisation From ECAT7:=
	    singles rates := Singles From Sgl File
		Singles Rates From Sgl File :=
		sgl_filename := ${sgl_filename}
		End Singles Rates From Sgl File:=
		normalisation_filename:= ${normfile}
	    End Bin Normalisation From ECAT7:=
	    time frame definition filename := ${scanname}.S
	    time frame number:= ${frame_num} 
	END:= 
EOF
        echo Executing ${CORRECT_PROJDATA} ${parfile}
	    ${CORRECT_PROJDATA} ${parfile}
        else
         echo Using existing file ${stir_sgl_prefix}_${acquisition_spec}.hs
        fi
   fi
done

set +e #continue on error such that we compare all files
trap "" ERR
all_ok=1
for frame_num in $allframes; do
    acquisition_spec=f${frame_num}g1d0b0
    echo ""
    echo Comparing results for frame ${frame_num}
    echo   Comparing data with singles from ECAT7 subheader
    # compare projdata corrected from ecat7 ( i.e. singles taken from ecat7) and bkproj_3D_lln
    compare_projdata  ${bkproj3d_prefix}_${acquisition_spec}_S_${acquisition_spec}.hs \
	${stir_prefix}_${acquisition_spec}.hs
    if [ $? != 0 ]; then
      all_ok=0
    fi
    if [ ${do_sgl_test} = 1 ]; then
        echo   Comparing data with singles from .sgl
	compare_projdata ${stir_sgl_prefix}_${acquisition_spec}.hs  ${stir_prefix}_${acquisition_spec}.hs
        if [ $? != 0 ]; then
         all_ok=0
        fi
    fi
done

if [ $all_ok = 1 ]; then
echo All tests ok
echo 'You can clean up by typing "rm -f *.par normalised*"'
echo 'but you might want to keep normalised_bkproj3d* to save some time in the future.'
else
echo Problems detected
fi

