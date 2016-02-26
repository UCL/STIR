# Demo of how to use STIR from python to display some images with matplotlib

# Copyright 2012-06-05 - 2013 Kris Thielemans

# This file is part of STIR.
#
# This file is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or
# (at your option) any later version.
#
# This file is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# See STIR/LICENSE.txt for details

#%% Initial imports
import numpy
import pylab
import stir
import stirextra

#%% You could set "interactive display" on
# This would mean that plots are generated and the program immediatelly continues,
# as opposed to waiting for you to close a figure.

#pylab.ion()

#%% image display

# read an image using STIR
image=stir.FloatVoxelsOnCartesianGrid.read_from_file('../../recon_test_pack/test_image_3.hv')

# convert data to numpy 3d array
npimage=stirextra.to_numpy(image);

# make some plots
# first a bitmap
pylab.figure()
pylab.imshow(npimage[10,:,:], cmap='hot');
pylab.title('slice 10 (starting from 0)')
pylab.show()
# now a profile
pylab.figure();
pylab.plot(npimage[10,45,:]);
pylab.xlabel('x')
pylab.title('slice 10, line 45 (starting from 0)')
pylab.show()


#%% Let's read some projection data

projdata=stir.ProjData.read_from_file('../recon_demo/smalllong.hs')
# get stack of all sinograms
sinos=stirextra.to_numpy(projdata);
# display a single sinogram
fig=pylab.figure();
hdl=pylab.imshow(sinos[10,:,:]);
pylab.title('sinogram 10 (starting from 0)');
# set some options as illustration how to use matplotlib
hdl.set_cmap('gray')
# change range of colormap
pylab.clim(0,sinos.max()*.9)
pylab.colorbar()
pylab.show()
