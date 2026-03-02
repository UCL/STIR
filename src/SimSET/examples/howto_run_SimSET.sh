#! /bin/sh

# This is an example run with the templates distributed with STIR (appropriate for the HR+).
# The code below works in bash, sh, ksh etc, but needs to be modified for csh.
# Authors: Kris Thielemans
#
#
#  Copyright (C) 2005 - 2006, Hammersmith Imanet Ltd
#  Copyright (C) 2011-07-01 - 2012, Kris Thielemans
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

# adjust location of SimSET to where you have it installed.
SIMSET_DIR=~/simset/2.9.1
export SIMSET_DIR

#All STIR utilities/scripts have to be in your path, e.g. if your 
#INSTALL_PREFIX was ~/STIR-bin, you could do
#   PATH=$PATH:~/STIR-bin/bin

# generate emission image
generate_image generate_uniform_cylinder.par 
# use the same size for the attenuation in this example
# above par file sets image values to 1, so line below will use water for attenuation
stir_math --including-first --times-scalar 0.096 uniform_cylinder_atten.hv my_uniform_cylinder.hv

# give the simulation a name. All output files will go into a new subdirectory of this name
SIM_NAME=sim3
# number of decays to simulate
PHOTONS=5000000
# specify names/locations of input files
EMISS_DATA=my_uniform_cylinder.hv
ATTEN_DATA=uniform_cylinder_atten.hv
templ_dir=`pwd`
TEMPLATE_PHG=${templ_dir}/template_phg.rec
TEMPLATE_BIN=${templ_dir}/template_bin.rec
TEMPLATE_DET=${templ_dir}/template_det.rec
# specify scanner
SCANNER="ECAT HR+"
# maximum ring difference to store in conversion from SimSET to Interfile projdata
NUM_SEG=31
# export all variables
export SIM_NAME EMISS_DATA ATTEN_DATA TEMPLATE_PHG TEMPLATE_BIN TEMPLATE_DET 
export PHOTONS NUM_SEG SCANNER

# set the simulation going
run_SimSET.sh
