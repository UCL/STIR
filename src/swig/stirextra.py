# A simple module with a few python functions to make it easier to work with STIR
# Copyright (C) 2012 Kris Thielemans
# Copyright (C) 2013 University College London

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
import sys
if sys.version_info < (3,):
    import exceptions

def get_bounding_box_indices(image):
    """
        return [min,max] tuple with indices of first and last voxel in a STIR image
        """
    num_dims = image.get_num_dimensions();
    if num_dims == 2:
        minind=stir.Int2BasicCoordinate()
        maxind=stir.Int2BasicCoordinate()
    elif num_dims == 3:
        minind=stir.Int3BasicCoordinate()
        maxind=stir.Int3BasicCoordinate()
    else:
        raise exceptions.NotImplementedError('need to handle dimensions different from 2 and 3')
    
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
    num_dims = image.get_num_dimensions();
    if num_dims == 2:
        [y,x]=numpy.mgrid[minphys[1]:maxphys[1]:sizes[1]*1j, minphys[2]:maxphys[2]:sizes[2]*1j];
        return [y,x]
    elif num_dims == 3:
        [z,y,x]=numpy.mgrid[minphys[1]:maxphys[1]:sizes[1]*1j, minphys[2]:maxphys[2]:sizes[2]*1j, minphys[3]:maxphys[3]:sizes[3]*1j];
        return [z,y,x]
    else:
        raise exceptions.NotImplementedError('need to handle dimensions different from 2 and 3')

def to_numpy(stirdata):
    """
        return the data in a STIR image or other Array as a numpy array
        """
    # construct a numpy array using the "flat" STIR iterator
    try:
        npstirdata=numpy.fromiter(stirdata.flat(), dtype=numpy.float32);
        # now reshape into ND array
        npdata=npstirdata.reshape(stirdata.shape());
        return npdata
    except:
        # hopefully it's projection data
        stirarray=stirdata.to_array();
        return to_numpy(stirarray);
