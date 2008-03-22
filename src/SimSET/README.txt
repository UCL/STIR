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

Available utilities:
(see code for more info)

- conv_to_SimSET_att_image
Allows converting an image with attenuation factor for 511 keV photons to a (8-bit)
index file that can be used as input for SimSET.

- conv_SimSET_projdata_to_STIR.sh
Allows converting SimSET output sinograms to STIR Interfile files.
