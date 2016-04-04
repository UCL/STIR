# Example script to run mMR sinograms data (need to have span1 delayeds)
# Author: Kris Thielemans

# input variables
PSINO=psino_span11.hs
NORM=Norm_20141010084929.n
export atnimg=cpmr-LM-00-umap.hv
export bedcoilatnimg=cpmr-LM-00-umap-hardware.hv

# create randoms estimate
INPUT=dsino_span1.hs TEMPLATE=${PSINO} howto_create_randoms_from_delayed_sino.sh 
convertSiemensInterfileToSTIR.sh  ${NORM}.hdr ${NORM}.hdr.STIR

export sino_input=${PSINO} 
export ECATNORM=${NORM}.hdr.STIR

convertSiemensInterfileToSTIR.sh ${atnimg} ${atnimg}.STIR
atnimg=${atnimg}.STIR
convertSiemensInterfileToSTIR.sh ${bedcoilatnimg} ${bedcoilatnimg}.STIR
bedcoilatnimg=${bedcoilatnimg}.STIR

howto_scatter_and_recon.sh
