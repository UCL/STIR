#! /bin/sh
# A script to check to see if reconstruction of simulated data gives the expected result.
#
#  Copyright (C) 2011, Hammersmith Imanet Ltd
#  Copyright (C) 2014, University College London
#  This file is part of STIR.
#
#  This file is free software; you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation; either version 2.1 of the License, or
#  (at your option) any later version.

#  This file is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  See STIR/LICENSE.txt for details
#      
# Authors:  Kris Thielemans
#           Matthew Strugari

echo This script should work with STIR version 5.1. If you have
echo a later version, you might have to update your test pack.
echo Please check the web site.
echo

# Store the input variables for calling with SPECTUB and PinholeSPECTUB scripts
INPUTVAR=$*

#
# Parse option arguments (--)
# Note that the -- is required to suppress interpretation of $1 as options 
# to expr
#
if test "$1" = "--help"; then
    echo "Usage: `basename $0` [--mpicmd somecmd] [--usecache] [install_dir]"
    echo "(where [] means that an argument is optional)"
    echo "Note: SPECT libraries are not yet configured for MPI capabilities."
    echo "See README.txt for more info."
    exit 1
fi

for SPECTtest in SPECTUB PinholeSPECTUB; do

    echo
    echo "********************************************************************************"
    echo "Changing to ${SPECTtest} directory and calling"
    echo "run_${SPECTtest}_tests.sh ${INPUTVAR}"
    echo "********************************************************************************"
    echo
    cd ${SPECTtest}
    ./run_${SPECTtest}_tests.sh ${INPUTVAR}
    if [ $? -ne 0 ]; then
       echo "Error running run_${SPECTtest}_tests.sh. Stopping with SPECT tests."
       exit 1
    fi
    
    # return to previous directory
    echo
    echo "Returning to previous directory..."
    cd -
done
