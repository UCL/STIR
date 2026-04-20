#! /bin/bash
# Shell script for testing SimSET support
# see README.txt
#
#  Copyright (C) 2019 University of Hull
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

# The SimSET scripts used in this test were adapted by the BrainVis_SimSET project.
# Original authors: 
# Aida Niñerola
# Gemma Salvadó
# Modified by:
# Jesús Silva
#
# Authors: Nikos Efthimiou
#         Jesus Silva

echo This script should work with STIR version ">"3.0. If you have
echo a later version, you might have to update your test pack.
echo Please check the web site.
echo

#
# Parse option arguments (--)
# Note that the -- is required to suppress interpretation of $1 as options 
# to expr
#
while test `expr -- "$1" : "--.*"` -gt 0
do

if test "$1" = "--help"
  then
    echo "First edit the run_SimSET_simulation_and_reconstruction.sh to point to then
    SimSET bin folder."
    echo "Usage: run_SimSET_simulation_and_reconstruction.sh [install_dir]"
    echo "See README.txt for more info."
    exit 1
  else
    echo Warning: Unknown option "$1"
    echo rerun with --help for more info.
    exit 1
  fi

  shift 1

done 

if [ $# -eq 1 ]; then
  echo "Prepending $1 to your PATH for the duration of this script."
  PATH=$1:$PATH
fi

##########################################
#                                        #
#     !EDIT THIS TO THE SIMSET PATH!     #
#                                        #
DIR_SIMSET=/home/nikos/Workspace/src/2.9.2
##########################################
export PATH=$PATH:$DIR_SIMSET/bin

#
# Run a short SimSET simulation
#
echo
echo -------- Uncompressing the data files ---------
echo

tar xvzf SimSET.tar.gz

#
# Run a short SimSET simulation
#
echo
echo -------- Running a short SimSET simulation ---------
echo

cp tmpl_phginitialParams_mCT.rec my_phginitialParams_mCT.rec
sed -i "s|DIR_SIMSET|${DIR_SIMSET}|g" "my_phginitialParams_mCT.rec"

phg my_phginitialParams_mCT.rec



if [ -z "${error_log_files}" ]; then
 echo "All tests OK!"
 echo "You can remove all output using \"rm -f my* && rm -f rec*\""
else
 echo "There were errors. Check ${error_log_files}"
fi
