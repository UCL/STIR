#! /bin/sh -e
# Example script to register MRAC/CTAC files from DICOM to a STIR NAC image, such that it is
# suitable for attenuation correction.
# 
# Relies on reg_aladin from NiftyReg, but you could use other registration algorithms of course
# Author: Kris Thielemans
# SPDX-License-Identifier: Apache-2.0
# See STIR/LICENSE.txt for details

# directory with some standard .par files
: ${pardir:=$(dirname $0)}

if [ $# -ne 3 ]; then
  echo "Usage: `basename $0` output_filename_prefix GE_AC_filename STIR_NAC_filename"
  echo 'This creates a Nifti file (called <output_filename_prefix>.nii) with the registered mu-map'
  exit 1
fi

output_filename_prefix=$1
GE_AC_filename=$2
NAC_filename=$3

# flip GE images to (wrong) output
invert_axis x ${output_filename_prefix}_prereg.hv ${GE_AC_filename}
invert_axis z ${output_filename_prefix}_prereg.hv ${output_filename_prefix}_prereg.hv

# convert to Nifti
stir_math --output-format $pardir/../samples/stir_math_ITK_output_file_format.par ${output_filename_prefix}_prereg.nii ${output_filename_prefix}_prereg.hv
NAC_filename_nii=${NAC_filename%%.*}_copy.nii
stir_math --output-format $pardir/../samples/stir_math_ITK_output_file_format.par ${NAC_filename_nii} ${NAC_filename}

# register
reg_aladin -ref ${NAC_filename_nii} -flo ${output_filename_prefix}_prereg.nii -res ${output_filename_prefix}.nii -rigOnly -speeeeed

