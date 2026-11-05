#! /bin/bash
#
# Converts a SimSET weight file from a block-detector simulation binned by
# crystal pair (bin_by_crystal = true) to STIR projection data, by calling
# conv_SimSET_crystal_pairs_to_STIR.
# The mode is read from the weight file header (with SimSET's printheader), and
# the block layout from the STIR template, which therefore has to describe the
# simulated detector.
#
# Supported SimSET binning:
#   scatter_param 4, min_s 0, max_s 1 -> <output_prefix>_unscattered.hs and
#                                        <output_prefix>_singles_scat.hs
#   scatter_param 5, min_s 2, max_s 2 -> <output_prefix>_multiples.hs
#
# SIMSET_DIR has to be set to your SimSET installation.
#
#  This file is part of STIR.
#
#  SPDX-License-Identifier: Apache-2.0
#
#  See STIR/LICENSE.txt for details

if [ $# -ne 3 ]; then
    echo "usage:"
    echo "$0 output_prefix template.hs simset-weight-file"
    exit 1
fi

output_prefix=$1
template=$2
weight_file=$3

set -e
script_name="$0"
trap "echo ERROR in script $script_name" ERR

if [ -z "${SIMSET_DIR}" ]; then
    echo "ERROR: SIMSET_DIR is not set. Set it to your SimSET installation, e.g."
    echo "  export SIMSET_DIR=~/simset/2.9.2"
    exit 1
fi
PRINTHEADER=${SIMSET_DIR}/bin/printheader

for f in "${template}" "${weight_file}"; do
  if [ ! -r "$f" ]; then
    echo "ERROR: cannot read $f"
    exit 1
  fi
done

# find the mode from the binning settings in the SimSET header
# (printheader can fail without a useful exit status, so check its output instead)
header=$(${PRINTHEADER} "${weight_file}" 2>/dev/null) || true
scatter_parameter=$(echo "$header" | grep "Binning: scatter parameter" | awk '{ print $4 }')
min_s=$(echo "$header" | grep "Binning: min number of scatters" | awk '{ print $6 }')
max_s=$(echo "$header" | grep "Binning: max number of scatters" | awk '{ print $6 }')
if [ -z "${scatter_parameter}" ]; then
    echo "ERROR: could not read the SimSET header of ${weight_file} with ${PRINTHEADER}."
    echo "Check that SIMSET_DIR is your SimSET installation and that this is a SimSET weight file."
    exit 1
fi

case "${scatter_parameter} ${min_s} ${max_s}" in
  "4 0 1") mode=singles_scat ;;
  "5 2 2") mode=multiples ;;
  *)
    echo "ERROR: ${weight_file} was binned with scatter_param ${scatter_parameter}, min_s ${min_s}, max_s ${max_s}."
    echo "Supported are scatter_param 4, min_s 0, max_s 1 (unscattered and scatter)"
    echo "and scatter_param 5, min_s 2, max_s 2 (multiple scatter)."
    exit 1
    ;;
esac

# find the block layout from the STIR template
template_value() {
  value=$(grep -i "^ *$1 *:=" "${template}" | head -1 | awk -F':=' '{ print $2 }' | tr -d ' \r')
  if [ -z "$value" ]; then
    echo "ERROR: \"$1\" not found in ${template}" >&2
    exit 1
  fi
  echo "$value"
}
num_rings=$(template_value "Number of rings")
num_detectors_per_ring=$(template_value "Number of detectors per ring")
crystals_per_block_axial=$(template_value "Number of crystals per block in axial direction")
crystals_per_block_transaxial=$(template_value "Number of crystals per block in transaxial direction")

num_block_rings=$(( num_rings / crystals_per_block_axial ))
num_blocks_per_ring=$(( num_detectors_per_ring / crystals_per_block_transaxial ))

conv_cmdline="conv_SimSET_crystal_pairs_to_STIR ${output_prefix} ${template} ${weight_file} ${mode} \
  ${num_block_rings} ${num_blocks_per_ring} ${crystals_per_block_axial} ${crystals_per_block_transaxial}"
echo Executing ${conv_cmdline}
${conv_cmdline}
