//
// $Id$
//
/*
    Copyright (C) 2001- $Date$, Hammersmith Imanet Ltd
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
\ingroup IO
\brief routines to convert AVW data structures to STIR
\author Kris Thielemans 

$Date$
$Revision$ 
*/

#include "stir/IO/stir_AVW.h"
#include "stir/IndexRange3D.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/CartesianCoordinate3D.h"


START_NAMESPACE_STIR

namespace AVW
{

template <typename elemT>
static 
void 
AVW_Volume_to_VoxelsOnCartesianGrid_help(VoxelsOnCartesianGrid<float>& image,
					 elemT const* avw_data,
					 const bool flip_z)
{
  // std::copy(avw_data, avw_data+avw_volume->VoxelsPerVolume, image->begin_all());
 
  // AVW data seems to be y-flipped
  for (int z=image.get_min_z(); z<=image.get_max_z(); ++z)
  {
    const int out_z =
      !flip_z ?  z : image.get_max_z() - z + image.get_min_z();
    for (int y=image.get_max_y(); y>=image.get_min_y(); --y)
    {
      for (int x=image.get_min_x(); x<=image.get_max_x(); ++x)
        image[out_z][y][x] = static_cast<float>(*avw_data++);
      //std::copy(avw_data, avw_data + image.get_x_size(), image[z][y].begin());
      //avw_data += image.get_x_size();
    }
  }
}


VoxelsOnCartesianGrid<float> *
AVW_Volume_to_VoxelsOnCartesianGrid(AVW_Volume const* const avw_volume,
					 const bool flip_z)
{
  // find sizes et al 

  const int size_x = avw_volume->Width;
  const int size_y = avw_volume->Height;
  const int size_z = avw_volume->Depth;
  IndexRange3D range(0, size_z-1,
                     -(size_y/2), -(size_y/2)+size_y-1,
                     -(size_x/2), -(size_x/2)+size_x-1);

  CartesianCoordinate3D<float> voxel_size;
  voxel_size.x() = 
    static_cast<float>(AVW_GetNumericInfo("VoxelWidth", avw_volume->Info));
  if (voxel_size.x()==0)
  {
    warning("AVW_Volume_to_VoxelsOnCartesianGrid: VoxelWidth not found or 0");
  }
  
  voxel_size.y() = 
    static_cast<float>(AVW_GetNumericInfo("VoxelHeight", avw_volume->Info));
  if (voxel_size.y()==0)
  {
    warning("AVW_Volume_to_VoxelsOnCartesianGrid: VoxelHeight not found or 0");
  }
  
  voxel_size.z() = 
    static_cast<float>(AVW_GetNumericInfo("VoxelDepth", avw_volume->Info));
  if (voxel_size.z()==0)
  {
    warning("AVW_Volume_to_VoxelsOnCartesianGrid: VoxelDepth not found or 0");
  }

  // construct VoxelsOnCartesianGrid
  VoxelsOnCartesianGrid<float> * volume_ptr =
    new VoxelsOnCartesianGrid<float>(range, 
                                     CartesianCoordinate3D<float>(0,0,0),
                                     voxel_size);

  // fill in data 
  switch(avw_volume->DataType)
  {
  case AVW_SIGNED_CHAR:
    {
      AVW_Volume_to_VoxelsOnCartesianGrid_help(*volume_ptr, 
					       reinterpret_cast<signed char const *>(avw_volume->Mem), flip_z);      
      break;
    }
  case AVW_UNSIGNED_CHAR:
    {
      AVW_Volume_to_VoxelsOnCartesianGrid_help(*volume_ptr, 
					       reinterpret_cast<unsigned char const *>(avw_volume->Mem), flip_z);
      break;
    }
  case AVW_UNSIGNED_SHORT:
    {
      AVW_Volume_to_VoxelsOnCartesianGrid_help(*volume_ptr, 
					       reinterpret_cast<unsigned short const *>(avw_volume->Mem), flip_z);
      break;
    }
  case AVW_SIGNED_SHORT:
    {
      AVW_Volume_to_VoxelsOnCartesianGrid_help(*volume_ptr,
					       reinterpret_cast<signed short const *>(avw_volume->Mem), flip_z);
      break;
    }
  case AVW_UNSIGNED_INT:
    {
      AVW_Volume_to_VoxelsOnCartesianGrid_help(*volume_ptr,
					       reinterpret_cast<unsigned int const *>(avw_volume->Mem), flip_z);
      break;
    }
  case AVW_SIGNED_INT:
    {
      AVW_Volume_to_VoxelsOnCartesianGrid_help(*volume_ptr,
					       reinterpret_cast<signed int const *>(avw_volume->Mem), flip_z);
      break;
    }
  case AVW_FLOAT:
    {
      AVW_Volume_to_VoxelsOnCartesianGrid_help(*volume_ptr, 
					       reinterpret_cast<float const *>(avw_volume->Mem), flip_z);
      break;
    }
  default:
    {
      warning("AVW_Volume_to_VoxelsOnCartesianGrid: unsupported data type %d\n",
        avw_volume->DataType);
      return 0;
    }
  }
         
  return volume_ptr;
}

} // end namespace AVW

END_NAMESPACE_STIR

