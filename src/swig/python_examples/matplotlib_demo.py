# Demo of how to use STIR from python to display some images with matplotlib

# Copyright 2012-06-05 - $Date$ Kris Thielemans

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

import numpy
import pylab
import stir
import stirextra

# read an image using STIR
image=stir.FloatVoxelsOnCartesianGrid.read_from_file('../../recon_test_pack/test_image_3.hv')

# convert data to numpy 3d array
npimage=stirextra.to_numpy(image);

# make some plots
# first a bitmap
pylab.figure()
pylab.imshow(npimage[10,:,:]);
pylab.title('slice 10 (starting from 0)')
pylab.show()
# now a profile
pylab.figure()
pylab.plot(npimage[10,45,:]);
pylab.xlabel('x')
pylab.title('slice 10, line 45 (starting from 0)')
pylab.show()
