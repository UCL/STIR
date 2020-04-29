#!/bin/bash

function print_usage() {
	echo "Usage: $0 [-h|--help] RAW_DATA_FOLDER TSTART TSTOP"
}

# Print full help
if [[ $1 == "-h" || $1 == "--help" ]]; then
	print_usage
	echo -e "\nThis bash script compares the STIR-wrapped NiftyPET functionality to NiftyPET.\n"
	echo -e "### Test content\n"
	echo -e "From raw dicom data (RAW_DATA_FOLDER) and a given time window (TSTART and TSTOP), prompt sinograms are extracted with NiftyPET, STIR-NiftyPET and STIR. Randoms are estimated, and for NiftyPET and STIR-NiftyPET, a norm sinogram is extracted.\n"
	echo -e "Forward and back projections are performed with NiftyPET and STIR-NiftyPET with no corrections.\n"
	echo -e "At each step, the results are compared to ensure consistency.\n"
	echo -e "Python2 is required for NiftyPET, so NIFTYPET_PYTHON_EXECUTABLE should be set as an environmental variable.\n"
	echo -e "### Data\n"
	echo -e "An example of test data can be found here - [https://doi.org/10.5281/zenodo.1472951](https://doi.org/10.5281/zenodo.1472951).\n"
	echo -e "Since the data needs to be read by NiftyPET, it should be in the raw form of .dcm/.bf, and not .l.hdr/.l.\n"
	exit 0
fi

# Check input arguments
if [[ "$#" -ne 3 ]]; then
	print_usage
	exit 1
fi
NTYP_RAW_DATA_FOLDER=$1
SINO_TSTART=$2
SINO_TSTOP=$3

# Check for python versions
if [[ $NIFTYPET_PYTHON_EXECUTABLE == "" ]]; then
	echo "Please set environmental variable: NIFTYPET_PYTHON_EXECUTABLE"
	exit 1
fi
NP_PY_EXE=$NIFTYPET_PYTHON_EXECUTABLE

fail_count=0

function check_result() {
	res=$2
	if [ $res -eq 99 ]; then 
		fail_count=$((fail_count+1)); echo "Test failed. Fail count=$fail_count"
	elif [ $res -ne 0 ]; then 
		echo "Exiting at line $1"; exit $res
	fi
}

########################################################################################
#
#    COMPARE SINOGAMS
#
########################################################################################

compare_sinos_np_space() {
	echo -e "\nComparing sinograms in NiftyPET space..."
	sino_np=$1
	sino_stir=$2
	sino_stir_2_np=NP_${sino_stir%.*}.dat
	conv_NiftyPET_stir $sino_stir_2_np $sino_stir sinogram toNP
	res=$?; if [ $res -ne 0 ]; then echo "Exiting at line $LINENO"; exit $res; fi
	$NP_PY_EXE -c \
"import sys
import numpy as np
f_sino1=sys.argv[1]
f_sino2=sys.argv[2]
sino1 = np.fromfile(f_sino1, dtype='float32')
sino2 = np.fromfile(f_sino2, dtype='float32')
diff = sino1 - sino2
norm = np.linalg.norm(diff.flatten()) / diff.size
print('max sino STIR->NP = ' + str(np.abs(sino1).max()))
print('max sino NP = ' + str(np.abs(sino2).max()))
print('max sino difference = ' + str(np.abs(diff).max()))
print('difference norm = ' + str(norm))
if norm > 1:
	sys.exit(99)" \
		$sino_stir_2_np $sino_np
	check_result $LINENO $?
}

########################################################################################
#
#    COMPARE IMAGES
#
########################################################################################

compare_ims_np_space() {
	echo -e "\nComparing images in NiftyPET space..."
	im_np=$1
	im_stir=$2
	im_stir_2_np=NP_${im_stir%.*}.dat
	conv_NiftyPET_stir $im_stir_2_np $im_stir image toNP
	res=$?; if [ $res -ne 0 ]; then echo "Exiting at line $LINENO"; exit $res; fi
	$NP_PY_EXE -c \
"import sys
import numpy as np
f_im1=sys.argv[1]
f_im2=sys.argv[2]
im1 = np.fromfile(f_im1, dtype='float32')
im2 = np.fromfile(f_im2, dtype='float32')
diff = im1 - im2
norm = np.linalg.norm(diff.flatten()) / diff.size
print('max im STIR->NP = ' + str(np.abs(im1).max()))
print('max im NP = ' + str(np.abs(im1).max()))
print('max im difference = ' + str(np.abs(diff).max()))
print('difference norm = ' + str(norm))
if norm > 1:
	sys.exit(99)" \
		$im_stir_2_np $im_np
	check_result $LINENO $?
}



########################################################################################
#
#    PROJECTION WITH NP
#
########################################################################################

project_np() {
	echo -e "\nProjecting with NiftyPET"
	$NP_PY_EXE -c \
"import sys
from niftypet import nipet
import numpy as np
mMRpars = nipet.get_mmrparams()
f_out=sys.argv[1]
f_in=sys.argv[2]
type=sys.argv[3]

if type=='fwd':
	im = np.fromfile(f_in, dtype='float32')
	im = np.reshape(im, (127, 320, 320))
	im = np.transpose(im, (1, 2, 0))
	out = nipet.frwd_prj(im, mMRpars)
else:
	sino = np.fromfile(f_in, dtype='float32')
	sino = np.reshape(sino, (837, 252, 344))
	out = nipet.back_prj(sino, mMRpars)
	out = nipet.img.mmrimg.convert2dev(out, mMRpars['Cnt'])
	out = np.transpose(out, (2, 0, 1))

out.astype('float32').tofile(f_out)" \
		"$@"
	res=$?; if [ $res -ne 0 ]; then echo "Exiting at line $LINENO"; exit $res; fi
}

########################################################################################
#
#    PROJECTION WITH STIR
#
########################################################################################

function project_stir() {
	echo -e "\nProjecting with STIR's NiftyPET wrapper"
	if [[ $4 == "fwd" ]]; then
		echo -e "Forward Projector parameters:=
		type := NiftyPET
		forward projector using niftypet parameters:=
		end forward projector using niftypet parameters:=
		End:=" > $4.par
		forward_project $1 $2 $3 $4.par
		res=$?; if [ $res -ne 0 ]; then echo "Exiting at line $LINENO"; exit $res; fi
	elif [[ $4 == "bck" ]]; then
		echo -e "Back Projector parameters:=
		type := NiftyPET
		back projector using niftypet parameters:=
		end back projector using niftypet parameters:=
		End:=" > $4.par
		back_project $1 $2 $3 $4.par
		res=$?; if [ $res -ne 0 ]; then echo "Exiting at line $LINENO"; exit $res; fi
	fi
}



echo "########################################################################################"
echo "#                                                                                      #"
echo "#    GETTING FILENAMES...                                                              #"
echo "#                                                                                      #"
echo "########################################################################################"

$NP_PY_EXE -c \
"import sys
from niftypet import nipet
mMRpars = nipet.get_mmrparams()
datain = nipet.classify_input(sys.argv[1], mMRpars)
if not all(k in datain for k in ('lm_dcm','lm_bf','nrm_dcm')):
	raise AssertionError('Missing some input data. Example data: https://doi.org/10.5281/zenodo.1472951.')
f = open('LM_DCM.txt','w+')
f.write('LM_DCM=' + datain['lm_dcm'] + '\n')
f.write('LM_BF=' + datain['lm_bf'] + '\n')
f.write('NORM_DCM=' + datain['nrm_dcm'] + '\n')
f.write('NORM_BF=' + datain['nrm_bf'] + '\n')
f.close()" \
	$NTYP_RAW_DATA_FOLDER
res=$?; if [ $res -ne 0 ]; then echo "Exiting at line $LINENO"; exit $res; fi
source LM_DCM.txt
rm LM_DCM.txt
if [ ! -f STIR_lm.l.hdr ]; then
	echo "Converting DICOM listmode to interfile..."
	nm_extract -i $LM_DCM -o . -p STIR_lm && \
		sed -i "s#STIR_lm.l#${LM_BF}#g" "STIR_lm.l.hdr" && \
		rm STIR_lm.l
	res=$?; if [ $res -ne 0 ]; then echo "Exiting at line $LINENO"; exit $res; fi
fi
if [ ! -f STIR_norm.n.hdr ]; then
	echo "Converting DICOM listmode to interfile..."
	nm_extract -i $NORM_DCM -o . -p STIR_norm
	res=$?; if [ $res -ne 0 ]; then echo "Exiting at line $LINENO"; exit $res; fi
fi




echo "########################################################################################"
echo "#                                                                                      #"
echo "#    LISTMODE EXTRACTION                                                               #"
echo "#                                                                                      #"
echo "########################################################################################"

# Extract NP sinogram
if [ ! -f NP_sino.dat ]; then
	echo "Extracting sinogram with NiftyPET..."
	$NP_PY_EXE -c \
"import sys
import shutil
folderin=sys.argv[1]
tstart=int(sys.argv[2])
tstop=int(sys.argv[3])
from niftypet import nipet
mMRpars = nipet.get_mmrparams()
datain = nipet.classify_input(folderin, mMRpars)
hst = nipet.mmrhist(datain, mMRpars, t0=tstart, t1=tstop)
hst['psino'].astype('float32').tofile('NP_sino.dat')
rands = nipet.randoms(hst, mMRpars)[0]
rands.astype('float32').tofile('NP_rands.dat')
norm = nipet.mmrnorm.get_norm_sino(datain, mMRpars, hst)
norm.astype('float32').tofile('NP_norm.dat')
mu_o = nipet.obj_mumap(datain, mMRpars, outpath='.', store=False)['im']
attn = nipet.frwd_prj(mu_o, mMRpars, attenuation=True)
attn.astype('float32').tofile('NP_attn.dat')
shutil.rmtree('mumap-obj')
" \
	$NTYP_RAW_DATA_FOLDER $SINO_TSTART $SINO_TSTOP
	res=$?; if [ $res -ne 0 ]; then echo "Exiting at line $LINENO"; exit $res; fi
	conv_NiftyPET_stir NP_attn_as_STIR NP_attn.dat sinogram toSTIR 
	res=$?; if [ $res -ne 0 ]; then echo "Exiting at line $LINENO"; exit $res; fi
fi

# Extract with STIR's NiftyPET wrapper
if [ ! -f STIR_sino.hs ]; then
	echo "Extracting sinogram with STIR's NiftyPET wrapper..."
	lm_to_projdata_NiftyPET $LM_BF $SINO_TSTART $SINO_TSTOP -N $NORM_BF -p STIR_sino -r STIR_rands -n STIR_norm
	res=$?; if [ $res -ne 0 ]; then echo "Exiting at line $LINENO"; exit $res; fi
fi



echo "########################################################################################"
echo "#                                                                                      #"
echo "#    COMPARE LISTMODE EXTRACTION AND SINOGRAM CONVERSION                               #"
echo "#                                                                                      #"
echo "########################################################################################"

compare_sinos_np_space NP_sino.dat STIR_sino.hs




echo "########################################################################################"
echo "#                                                                                      #"
echo "#    RANDOMS COMPARISON                                                                #"
echo "#                                                                                      #"
echo "########################################################################################"

compare_sinos_np_space NP_rands.dat STIR_rands.hs




echo "########################################################################################"
echo "#                                                                                      #"
echo "#    NORM COMPARISON                                                                   #"
echo "#                                                                                      #"
echo "########################################################################################"

compare_sinos_np_space NP_norm.dat STIR_norm.hs




echo "########################################################################################"
echo "#                                                                                      #"
echo "#    PROJECT WITH NIFTYPET                                                             #"
echo "#                                                                                      #"
echo "########################################################################################"

project_np NP_im.dat NP_sino.dat bck
project_np NP_fwd.dat NP_im.dat fwd


echo "########################################################################################"
echo "#                                                                                      #"
echo "#    PROJECT WITH STIR                                                                 #"
echo "#                                                                                      #"
echo "########################################################################################"

conv_NiftyPET_stir NP_sino_as_STIR NP_sino.dat sinogram toSTIR
conv_NiftyPET_stir NP_im_as_STIR NP_im.dat image toSTIR

project_stir STIR_im.hv NP_sino_as_STIR.hs NP_im_as_STIR.hv bck
project_stir STIR_fwd.hs STIR_im.hv NP_sino_as_STIR.hs fwd 




echo "########################################################################################"
echo "#                                                                                      #"
echo "#    COMPARE PROJECTIONS                                                               #"
echo "#                                                                                      #"
echo "########################################################################################"

compare_ims_np_space NP_im.dat STIR_im.hv
compare_sinos_np_space NP_fwd.dat STIR_fwd.hs




echo "########################################################################################"
echo "#                                                                                      #"
echo "#    RESULTS                                                                           #"
echo "#                                                                                      #"
echo "########################################################################################"

rm tmp_* eff_out.txt geom_out.txt inter_out.txt 2> /dev/null

if [ $fail_count -eq 0 ]; then
	echo -e "\nFinished with no errors!"
	exit 0
else
	echo -e "\nFinished with $fail_count errors..."
	exit 1
fi