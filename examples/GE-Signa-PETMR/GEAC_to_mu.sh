#! /bin/bash -e

# Example script to create a mu-file in STIR units from a CTAC
# could be adapted for MRAC if slopes are adjusted
# Author: Kris Thielemans
# SPDX-License-Identifier: Apache-2.0
# See STIR/LICENSE.txt for details

if [ $# -ne 2 ]; then
  echo "Usage: `basename $0` output_filename CT_filename"
  echo "This creates a mu-map from GE-CT (preliminary!)"
  exit 1
fi

# directory with some standard .par files
: ${pardir:=$(dirname $0)}

# This default will not work for most. Adjust!
: ${CT_SLOPES_FILENAME:=$STIR_install/share/stir/ct_slopes.json}
if [ ! -r "$CT_SLOPES_FILENAME" ]; then
    echo "You need to set the CT_SLOPES_FILENAME environment variable to an existing file."
    exit 1
fi
# note: variable name is used in .par
export CT_SLOPES_FILENAME

# TODO get kV from the CT dicom header
kV=120
export kV

# STIR 5 supports a filter that can both do the slopes and Gaussian
# postfilter "$1" "$2"  "${pardir}/GE_HUToMu.par"
ctac_to_mu_values  -o "$1" -i "$2" -j $CT_SLOPES_FILENAME  -m GE -v $kV -k 511

# potentially postfilter with a Gaussian (usually required!)
