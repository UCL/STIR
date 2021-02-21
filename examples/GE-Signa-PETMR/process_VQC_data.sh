#! /bin/bash -e
# An example script to process the VQC data (which has to be downloaded first).
# This data is special as it is just 5 point sources. We don't have an MRAC for this data-set.
# To illustrate the use the full scripts, we create a "fake" attenuation image.
# Aside from attenution processing, this should work for other data in GE RDF9 as well.
#
# To extract the RDF files from GE DICOM, we rely on nm_extract from https://github.com/UCL/pet-rd-tools
#
# Author: Kris Thielemans

# By default, it will find the parameter files in the pardir directory below.
# You can change the file, or run the script by doing something like
#    pardir=~/wherever/
#    export pardir
#    $pardir/process_VQC_data.sh

: ${pardir:=~/devel/STIR/examples/GE-Signa-PETMR/}
export pardir

#list_lm_events  --num-events-to-list 80 --coincidence 1 LIST0000.BLF > 80records.txt

mkdir output
cd output

listmode=../VQC_Phantom_Dataset_Share/LST/LST_30501_PET_Scan_for_VQC_Verification/LIST0000.BLF

# make a frame definition file with 1 frame for all the data
create_fdef_from_listmode.sh frames.fdef "$listmode"

# create prompts and randoms sinograms
INPUT="$listmode" FRAMES=frames.fdef $pardir/unlist_and_randoms.sh

# Create a dummy attenuation file (we don't have one for the VQC phantom)
generate_image $pardir/generate_zero_atn.par
atnimg=zero_atn.hv
export atnimg

# Extract RDFs such as norm factors (and others) from DICOM
for f in ../VQC_Phantom_Dataset_Share/PTRAW/30001_PET_Scan_for_VQC_Verification/*img; do
    nm_extract -i $f -o ./;
done

# Find the norm file
RDFNORM=`ls *norm.rdf`
export RDFNORM

# Run scatter estimation and OSEM
$pardir/scatter_and_recon.sh




