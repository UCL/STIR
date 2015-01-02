#! /bin/sh
# This script is part of the example suite on how to run simulation.
# It generates emission and attenuation images and a template projection
# data (which will be used for forward projection).
#
# Note: these are examples only. Modify this for your own needs.
#
#  Copyright (C) 2013-2014 University College London
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

# first need to set this to the C locale, as this is what the STIR utilities use
# otherwise, awk might interpret floating point numbers incorrectly
LC_ALL=C
export LC_ALL

mkdir -p output
cd output

echo "===  make emission image"
generate_image  ../generate_uniform_cylinder.par
echo "===  make attenuation image"
generate_image  ../generate_attenuation_image.par
# Alternative to illustrate how to use the emission image a template for
# attenuation (every voxel with some non-zero emission data is set to
# attenuation of water)
# stir_math --including-first --times-scalar .096 my_atten_image.hv my_uniform_cylinder.hv

echo "===  create template sinogram (DSTE in 3D with max ring diff 1 to save time)"
# Note: the following uses some fancy shell-scripting syntax to be able to
# run this script automatically. You probably just want to run
# create_projdata_template without input redirection and answer the questions
# interacitvely.
template_sino=my_DSTE_3D_rd1_template.hs
cat > my_input.txt <<EOF
Discovery STE
1
n

0
1
EOF
create_projdata_template  ${template_sino} < my_input.txt > my_create_${template_sino}.log 2>&1
if [ $? -ne 0 ]; then 
  echo "ERROR running create_projdata_template. Check my_create_${template_sino}.log"; exit 1; 
fi

# compute ROI values (as illustration)
input_image=my_uniform_cylinder.hv
ROI=../ROI_uniform_cylinder.par
list_ROI_values ${input_image}.roistats ${input_image} ${ROI} 0 > /dev/null 2>&1
input_ROI_mean=`awk 'NR>2 {print $2}' ${input_image}.roistats`

cd ..

