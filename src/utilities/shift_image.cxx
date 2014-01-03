/*
 Copyright (C) 2010 - 2013, King's College London
 This file is part of STIR.
 
 This file is free software; you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation; either version 2.3 of the License, or
 (at your option) any later version.
 
 This file is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Lesser General Public License for more details.
 
 See STIR/LICENSE.txt for details
 */
/*!
 \file 
 \ingroup utilities
 \brief This program shifts the origin of an image.
 \author Charalampos Tsoumpas
*/
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/Succeeded.h"
#include "stir/round.h"

#include <math.h>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
#endif

USING_NAMESPACE_STIR


int main(int argc, char **argv)
{
  if(argc<3 || argc>7) {
    cerr<<"Usage: " << argv[0] << " <output image filename> <input image filename> [x-shift] [y-shift] [z-shift] [extend_borders]\n"
        << "all shifts are in mm and defaults are set to 0mm\n"
        << "extend borders is either 1 or 0, defaults to 0\n";
    exit(EXIT_FAILURE);
  }
  
  // get parameters from command line
  char const * const output_filename = argv[1];
  char const * const input_filename = argv[2];
  const float x_shift_in_mm = (argc>3) ? static_cast<float>(atof(argv[3])) : 0;
  const float y_shift_in_mm = (argc>4) ? static_cast<float>(atof(argv[4])) : 0;
  const float z_shift_in_mm = (argc>5) ? static_cast<float>(atof(argv[5])) : 0;
  const int extend_borders = (argc>6) ? atoi(argv[6]) : 0;
	
  // read image
  const shared_ptr<DiscretisedDensity<3,float> > density_sptr(
							      DiscretisedDensity<3,float>::read_from_file(input_filename));
  const DiscretisedDensityOnCartesianGrid <3,float>*  density_cartesian_sptr = 
    dynamic_cast< DiscretisedDensityOnCartesianGrid<3,float>* > (density_sptr.get());
  const BasicCoordinate<3,float> grid_spacing=density_cartesian_sptr->get_grid_spacing();
  const CartesianCoordinate3D<float> origin=density_cartesian_sptr->get_origin(); 

  const Coordinate3D<float> image_shift(z_shift_in_mm,y_shift_in_mm,x_shift_in_mm);
  const int voxel_shift_z=round(image_shift[1]/grid_spacing[1]);
  const int voxel_shift_y=round(image_shift[2]/grid_spacing[2]);
  const int voxel_shift_x=round(image_shift[3]/grid_spacing[3]);
  const Coordinate3D<int> num_voxels_to_shift(voxel_shift_z,voxel_shift_y,voxel_shift_x);
	
  const float actual_z_shift_in_mm = static_cast<float> (voxel_shift_z)*grid_spacing[1];
  const float actual_y_shift_in_mm = static_cast<float> (voxel_shift_y)*grid_spacing[2];
  const float actual_x_shift_in_mm = static_cast<float> (voxel_shift_x)*grid_spacing[3];
	
  std::cerr << "Actual z shift: " << actual_z_shift_in_mm << "mm\n";
  std::cerr << "Actual y shift: " << actual_y_shift_in_mm << "mm\n";
  std::cerr << "Actual x shift: " << actual_x_shift_in_mm << "mm\n";

  BasicCoordinate<3,int> min;	BasicCoordinate<3,int> max;
  const IndexRange<3> range=density_sptr->get_index_range();
  if (!range.get_regular_range(min,max))
    error("image is not in regular grid.\n");
  BasicCoordinate<3,int> out_min=min;	BasicCoordinate<3,int> out_max=max;	
  if (extend_borders==1) {
    out_min[1]-=abs(voxel_shift_z) ;
    out_min[2]-=abs(voxel_shift_y) ;
    out_min[3]-=abs(voxel_shift_x) ;
    out_max[1]+=abs(voxel_shift_z) ;
    out_max[2]+=abs(voxel_shift_y) ;
    out_max[3]+=abs(voxel_shift_x) ;
  }
  const IndexRange<3> out_range(out_min,out_max);
  VoxelsOnCartesianGrid<float> out_density(out_range,origin,grid_spacing);
	
  BasicCoordinate<3,int> c, d;
  for (c[1]=min[1]; c[1]<=max[1]; ++c[1])
    for (c[2]=min[2]; c[2]<=max[2]; ++c[2])
      for (c[3]=min[3]; c[3]<=max[3]; ++c[3])
        {
          d=c+num_voxels_to_shift;
          if (d[1]>=out_min[1] && d[2]>=out_min[2] && d[3]>=out_min[3] && d[1]<=out_max[1] && d[2]<=out_max[2] && d[3]<=out_max[3])
            out_density[d] = (*density_sptr)[c];
        }
	
  // write image
  Succeeded res = 
    OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
    write_to_file(output_filename, out_density);
  return res==Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}

