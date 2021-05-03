#!/bin/sh
# Takes an interfile projdata header, and creates a new one with view-offset set to 0.
#
#
#  Copyright (C) 2020, University College London
#  This file is part of STIR.
#
#  SPDX-License-Identifier: Apache-2.0
#
#  See STIR/LICENSE.txt for details
#
# Author: Kris Thielemans

if [ $# -ne 2 ]; then
  echo "Usage: `basename $0` output_filename projdata_interfile_header_filename"
  echo "sets view-offset to zero"
  echo "Warning: this script is unsafe. It doesn't do any checks at all."
  exit 1
fi

out_filename=$1
in_filename=$2

set -e
sed -e "s/View offset (degrees).*/View offset (degrees) := 0/" $in_filename > $out_filename
