#! /bin/bash -e

# Example script to create a mu-file in STIR units from a CTAC
# could be adapted for MRAC if slopes are adjusted
# Author: Kris Thielemans
# SPDX-License-Identifier: Apache-2.0
# See STIR/LICENSE.txt for details

if [ $# -ne 3 ]; then
  echo "Usage: `basename $0` output_filename CT_filename"
  echo "This creates a projdata with the normalisation factors"
  exit 1
fi

# directory with some standard .par files
: ${pardir:=$(dirname $0)}

# This default will not work for most. Adjust!
: {CT_SLOPES_FILENAME:=$STIR_install/../../stir/ct_slopes.json}
# note: variable name is used in .par
export CT_SLOPES_FILENAME
# TODO get kV from the CT dicom header
kV=120
export kV
postfilter ctac.hv  "${CT}"  "${pardir}/GE_HUToMu.par"
