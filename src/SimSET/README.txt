/* $Id$

    Copyright (C) 2008- $Date$, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/

This directory contains a set of utilities and scripts to make it easier to
use STIR together with SimSET.

WARNING: SimSET has miriad options. The files here assume that you run SimSET in 
a certain way. There are hardly any checks if this was the case or not.

WARNING: An important caveat is that STIR can at presently not handle SimSET data 
with an even number of tangential positions ('num_td_bins') with a symmetric range 
for min_td and max_td. This is because STIR uses a convention suitable for ECAT data 
taking interleaving of sinogram-bins into account. You will get artifacts in the 
images unless you use an odd number for num_td_bins.
Similarly, usually in SimSET you put the centre of the image in the centre of the 
scanner (although you don't have to of course). The current version of STIR will 
only do this if you have an odd number of pixels in x and y.

Most scripts require bash. Some require python. We also use various standard 
utilities such as awk, grep, tr.

Available utilities:
(see code for more info)

- conv_to_SimSET_att_image
Allows converting an image with attenuation factor for 511 keV photons to a (8-bit)
index file that can be used as input for SimSET.

- conv_SimSET_projdata_to_STIR.sh
Allows converting SimSET output sinograms ("weight files" to STIR Interfile files.
(Do not use the executable of the same name (without .sh) unless in emergencies).
Read the header of the script for some information.

- SimSET_STIR_names.sh
Prints the (root of) the names of the projection data constructed by the above script.

- make_hv_from_Simset_params.sh
Construct an Interfile header for a binary (attenuation or emission) image
output by SimSET.

- write_phg_image_info
Helps constructing a PHG input file by generating the object-spec.
You would not normally run this directly, but use stir_image_to_simset_object.sh

- stir_image_to_simset_object.sh
Helps constructing a PHG input file by generating the object-spec for an image
(as interpreted by list_image_info).
You probably don't need this if you use run_SimSET.sh

- run_SimSET.sh
A script to run SimSET. It takes input images that STIR can read, and a few templates
input files.

How to use
-----------
You first have to tell these routines where your SimSET installation is located, for instance

    SIMSET_DIR=~/simset/2.9.1
    export SIMSET_DIR

All STIR utilities/scripts have to be in your path, e.g. if your 
INSTALL_PREFIX was ~/STIR-bin:

    PATH=$PATH:~/STIR-bin/bin

See the SimSET/examples directory for an example.
