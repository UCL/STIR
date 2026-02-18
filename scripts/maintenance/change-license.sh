#! /bin/sh
# This script runs sed to change the GNU license to Apache 2.0
# Usage:
#    change-license.sh <some-path>
# This will change all files in that path and sub-folders.
# <some-path> can be replaced by arguments to "find" to limit the scope.
#
# The actual replacement is done with sed and the sed script change-license.sed.
#
# Copyright (C) 2021 University College London
# SPDX-License-Identifier: Apache-2.0
# Author: Kris Thielemans

script_path=`dirname $0`
if [ $# -eq 0 ]; then
    echo "Please add arguments for the \'find\' command (normally just .)"
    exit 1
fi
find $* -type f -exec grep -q 'GNU.*General Public License' {} \; -exec sed -E -i -f $script_path/change-license.sed {} \;

find $* -path .git -prune \
     -o -name "*[xhlkc]" -type f  -exec grep -l PARAPET {} \; -exec sed -E -i -f $script_path/add-PARAPET-license.sed {} \;

find $* -path .git -prune \
     -o -name FourierRebinning.cxx -exec sed -E -i -e s/Apache-2.0/LGPL-2.1-or-later/ {} \;
