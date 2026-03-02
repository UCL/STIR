#! /bin/sh
# This script runs the whole simulation and 2 example reconstructions
#
#  Copyright (C) 2013 University College London
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
# Author Kris Thielemans
# 

# add current directory to the path
# (necessary to find simulate_scatter.sh for instance)
PATH=`pwd`:$PATH
generate_input_data.sh 

# create sinograms
# note: names and directory are hard-wired in the script above (but not in the simulate_data.sh script)
cd output
template_sino=my_DSTE_3D_rd1_template.hs
simulate_data.sh my_uniform_cylinder.hv my_atten_image.hv ${template_sino} ../scatter.par ../scatter_template.hs
if [ $? -ne 0 ]; then
  echo "Error running simulation"
  exit 1
fi

# add Poisson noise
poisson_noise my_noise my_prompts.hs 1 2

# run 2 reconstructions
echo "=== running OSMAPOSL"
recon_iterative.sh my_OSEM my_noise.hs OSMAPOSL ../OSMAPOSL_QP.par my_multfactors.hs my_additive_sinogram.hs 
echo "=== running FBP2D"
recon_FBP2D.sh my_fbp my_noise.hs ../FBP2D.par my_multfactors.hs my_additive_sinogram.hs 
# amide my_OSEM_14.hv my_OSEM_28.hv my_fbp.hv 

echo "All done!"
echo "All files are in the 'output' directory"
