# A simple module with a few python functions to make it easier to work with STIR
# Copyright 2012-10-05 - $Date$ Kris Thielemans

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
import stir

def get_bounding_box_indices(image):
    """
    return [min,max] tuple with indices of first and last voxel in a STIR image
    """
    minind=stir.Int3BasicCoordinate()
    maxind=stir.Int3BasicCoordinate()
    image.get_regular_range(minind, maxind);
    return [minind, maxind]

def get_physical_coordinates_for_bounding_box(image):
    """
    return physical coordinates (in mm) for the first and last voxel in a STIR image
    """
    [minind,maxind]=get_bounding_box_indices(image);
    minphys=image.get_physical_coordinates_for_indices(minind);
    maxphys=image.get_physical_coordinates_for_indices(maxind);
    return [minphys, maxphys]

def get_physical_coordinates_for_grid(image):
    """
    return [z,y,x] tuple of grid coordinates for a STIR image
    """
    [minind,maxind]=get_bounding_box_indices(image);
    sizes=maxind-minind+1;
    minphys=image.get_physical_coordinates_for_indices(minind);
    maxphys=image.get_physical_coordinates_for_indices(maxind);
    [z,y,x]=numpy.mgrid[minphys[1]:maxphys[1]:sizes[1]*1j, minphys[2]:maxphys[2]:sizes[2]*1j, minphys[3]:maxphys[3]:sizes[3]*1j];
    return [z,y,x]

def to_numpy(image):
    """
    return the data in a STIR image as a 3D numpy array
    """
    [minind,maxind]=get_bounding_box_indices(image);
    sizes=maxind-minind+1;
    # construct a numpy array using the "flat" STIR iterator
    npimage=numpy.fromiter(image.flat(), dtype=numpy.float32);
    # now reshape into 3D array
    npimage=npimage.reshape(sizes[1], sizes[2], sizes[3]);
    return npimage
