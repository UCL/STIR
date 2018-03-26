//
//
/*
    Copyright (C) 2006- 2012, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2.0 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!
  \file 
  \ingroup utilities
 
  \brief  This program lists basic image info

  \author Thielemans


  \warning It only supports stir::VoxelsOnCartesianGrid type of images.
*/

#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/stream.h"
#include "stir/Succeeded.h"

#include <memory>
#include <iostream>
USING_NAMESPACE_STIR

int main(int argc, char *argv[])
{
  if(argc!=2) 
    {
      std::cerr<<"\nUsage: list_image_info image_filename";
      return EXIT_FAILURE;
    }
  std::auto_ptr<VoxelsOnCartesianGrid<float> >image_aptr =
    std::auto_ptr<VoxelsOnCartesianGrid<float> >
    (dynamic_cast<VoxelsOnCartesianGrid<float> *>(
	DiscretisedDensity<3,float>::read_from_file(argv[1]))
    );

  BasicCoordinate<3,int> min_indices, max_indices;
  if (!image_aptr->get_regular_range(min_indices, max_indices))
    error("Non-regular range of coordinates. That's strange.\n");

  BasicCoordinate<3,float> edge_min_indices(min_indices), edge_max_indices(max_indices);
  edge_min_indices-= 0.5F;
  edge_max_indices+= 0.5F;

  std::cout << "\nOrigin in mm {z,y,x}    :" << image_aptr->get_origin()
	    << "\nVoxel-size in mm {z,y,x}:" << image_aptr->get_voxel_size()
	    << "\nMin_indices {z,y,x}     :" << min_indices
	    << "\nMax_indices {z,y,x}     :" << max_indices
	    << "\nNumber of voxels {z,y,x}:" << max_indices - min_indices + 1
	    << "\nPhysical coordinate of first index in mm {z,y,x} :" 
	    << image_aptr->get_physical_coordinates_for_indices(min_indices)
	    << "\nPhysical coordinate of last index in mm {z,y,x}  :" 
	    << image_aptr->get_physical_coordinates_for_indices(max_indices)
	    << "\nPhysical coordinate of first edge in mm {z,y,x} :" 
	    << image_aptr->get_physical_coordinates_for_indices(edge_min_indices)
	    << "\nPhysical coordinate of last edge in mm {z,y,x}  :" 
	    << image_aptr->get_physical_coordinates_for_indices(edge_max_indices)
	    << "\nImage min: " << image_aptr->find_min()
	    << "\nImage max: " << image_aptr->find_max()
	    << "\nImage sum: " << image_aptr->sum()
	    << std::endl;

  return EXIT_SUCCESS;
}
