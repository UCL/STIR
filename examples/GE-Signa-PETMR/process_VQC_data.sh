#! /bin/bash -e
# An example script to process the VQC data (which has to be downloaded first).
# This data is special as it is just 5 point sources. We don't have an MRAC for this data-set.
# To illustrate the use the full scripts, we create a "fake" attenuation image.
# Aside from attenution processing, this should work for other data in GE RDF9 as well.
#
# To extract the RDF files from GE DICOM, we rely on nm_extract from https://github.com/UCL/pet-rd-tools
#
# Author: Kris Thielemans
# SPDX-License-Identifier: Apache-2.0
# See STIR/LICENSE.txt for details

# By default, it will find the parameter files in the same directory as the script.
# You can change the file, or run the script by doing something like
#    pardir=~/wherever/
#    export pardir
#    $pardir/process_VQC_data.sh
# By default, it will find the extracted data in ./VQC_Phantom_Dataset_Share
# You can set an environment variable datadir to change that
#    datadir=~/whereever/I/downloaded/VQC_Phantom_Dataset_Share
#    export datadir

: ${pardir:=$(dirname $0)}
# convert to absolute path (assumes that it exists), from DVK's answer on stackexchange
pardir=`cd "$pardir";pwd`
export pardir

: ${datadir:=./VQC_Phantom_Dataset_Share}
if [ ! -r "$datadir" ]; then
    echo "No data in $datadir. Download data or export datadir."
    exit 1
fi
datadir=`cd "$datadir";pwd`

#list_lm_events  --num-events-to-list 80 --coincidence 1 LIST0000.BLF > 80records.txt

mkdir output
cd output

listmode="$datadir"/LST/LST_30501_PET_Scan_for_VQC_Verification/LIST0000uncompressed.BLF

# make a frame definition file with 1 frame for all the data
create_fdef_from_listmode.sh frames.fdef "$listmode"

# create prompts and randoms sinograms
INPUT="$listmode" FRAMES=frames.fdef $pardir/unlist_and_randoms.sh

# Extract RDFs such as norm factors (and others) from DICOM
for f in "$datadir"/PTRAW/30001_PET_Scan_for_VQC_Verification/*img; do
    # ignore error for WCC files by adding "&& true" (otherwise the script aborts)
    nm_extract -i "$f" -o ./ && true
done

# Find the norm file
RDFNORM=`ls *norm.rdf`
export RDFNORM

# Run a reconstruction without attenuation
$pardir/NAC_recon.sh

# we stop here
exit

# you could continue with an AC reconstruction, but then we need to
# create a dummy attenuation file (we don't have one for the VQC phantom)
generate_image $pardir/generate_zero_atn.par
atnimg=zero_atn.hv

# For normal data, you will have to convert your PIFA to a mu-map in units of cm^-1, postsmoothing it a bit

# register the ctac to the NAC
"$pardir"/register_GEAC.sh  mu mrac.hv NAC_image_42.hv
atnimg=mu.nii

export atnimg
# Run scatter estimation and OSEM
$pardir/scatter_and_recon.sh


