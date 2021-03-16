: ${datadir:=`pwd`}
#~/MyLaptop/University\ College\ London/inm.physics\ -\ Documents/General/phantoms/DMI/dual-phantom-from-GE/exam177_dualPhantom_DMI3R_CT_RAW_RDFv9
: ${RDF:=${datadir}/raw_HDF5_decompressed/rdf_f1b1.rdf}
: ${CT:=`ls "${datadir}/CTAC/"*.1.img`}

: ${STIR_install:=~/devel/install/share/doc/stir-5.0}
: ${pardir:=~/devel/STIR/examples/GE-PETCT/}
export pardir

for f in "${datadir}/raw_dcm/"*.3.img; do nm_extract -i "$f" -o .; done

randoms=randoms_`basename "${RDF%%.*}"`.hs
construct_randoms_from_GEsingles "${randoms}" "${RDF}"

# copy prompts to Interfile to get round a limitation of the current
# RDF reader (it doesn't have get_sinogram(), and is rather slow)
stir_math -s `basename "${RDF%%.rdf}"`.hs "${RDF}"
RDF=`basename "${RDF%%.rdf}"`.hs


CT_SLOPES_FILENAME=$STIR_install/../../stir/ct_slopes.json
export CT_SLOPES_FILENAME
# TODO get kV from the CT dicom header
kV=120
export kV
postfilter ctac.hv  "${CT}"  "${pardir}/GE_HUToMu.par"

RDFNORM=`ls *norm.rdf`
export RDFNORM

sino_input=${RDF} randoms3d=${randoms} RDFNORM=${RDFNORM} num_subsets=17 num_subiters=42 $pardir/NAC_recon.sh

$pardir/register_GEAC.sh  mu ctac.hv NAC_image_42.hv

sino_input=${RDF} randoms3d=${randoms} RDFNORM=${RDFNORM} atnimg=mu.nii num_subsets=17 num_subiters=140 $pardir/scatter_and_recon.sh
