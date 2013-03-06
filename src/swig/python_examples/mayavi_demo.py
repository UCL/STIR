# Demo of how to mayavi to display some STIR data

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
from mayavi import mlab

# read in an image using STIR
image=stir.FloatVoxelsOnCartesianGrid.read_from_file('../../../recon_test_pack/test_image_3.hv')

# convert data to numpy 3d array
npimage=stirextra.to_numpy(image);
# get grid coordinates in mm
[z,y,x]=stirextra.get_physical_coordinates_for_grid(image);

# create a mayavi figure
mlab.figure();
# add image as a scalar_field
im=mlab.pipeline.scalar_field(z,y,x,npimage)
# add a slice
mlab.pipeline.image_plane_widget(im,
                                 plane_orientation='x_axes',
                                 slice_index=10,
                                 )
# could add an orthogonal slice
#mlab.pipeline.image_plane_widget(im,
#                            plane_orientation='y_axes',
#                            slice_index=10,
#                        )
#mlab.outline()

# some preliminary code to add the FOV to the plot. currently not used
def cylinder():
    pi = numpy.pi
    cos = numpy.cos
    sin = numpy.sin
    r = 90;
    l = 30;
    dphi, dz = pi/250.0, l/2
    [phi,z] = numpy.mgrid[0:2*pi+dphi*1.5:dphi,0:l:dz]
    x = r*sin(phi)
    y = r*cos(phi)
    return mlab.mesh(x, y, z, colormap="bone",opacity=.2)

