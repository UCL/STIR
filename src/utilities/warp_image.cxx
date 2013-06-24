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
  \ingroup spatial_transformation
 
  \brief This program warps an image.
  \author Charalampos Tsoumpas
  $Date$
  $Revision$
*/
#include "stir/spatial_transformation/warp_image.h"
#include "stir/Succeeded.h"
#include "stir/IO/OutputFileFormat.h"

USING_NAMESPACE_STIR


int main(int argc, char **argv)
{
  if(argc<6 || argc>8) {
    cerr<<"Usage: " << argv[0] << " <output image filename> <input image filename> [x-motion-field] [y-motion-field] [z-motion-field] [spline_type] [extend_borders]\n"
        << "all shifts are in mm\n"
        << "x, y, z are in STIR conventions\n"
        << "extend borders is either 1 or 0, defaults to 0\n";
    exit(EXIT_FAILURE);
  }
  
  // get parameters from command line
  char const * const output_filename = argv[1];
  char const * const input_filename = argv[2];
  char const * const motion_x_filename = argv[3];
  char const * const motion_y_filename = argv[4];
  char const * const motion_z_filename = argv[5];
  const int interpolation_type = (argc==6) ? 3 : atoi(argv[6]);	
  const int extend_borders = (argc<=7) ? 0 : atoi(argv[7]);

  const BSpline::BSplineType spline_type = static_cast<BSpline::BSplineType> (interpolation_type);

  std::cerr << "Interpolating using with splines level: " << spline_type << "\n"; 
  // read image
  const shared_ptr<DiscretisedDensity<3,float> > density_sptr(DiscretisedDensity<3,float>::read_from_file(input_filename));
  const shared_ptr<DiscretisedDensity<3,float> > motion_x_sptr(
                                                               DiscretisedDensity<3,float>::read_from_file(motion_x_filename));
  const shared_ptr<DiscretisedDensity<3,float> > motion_y_sptr(
                                                               DiscretisedDensity<3,float>::read_from_file(motion_y_filename));
  const shared_ptr<DiscretisedDensity<3,float> > motion_z_sptr(
                                                               DiscretisedDensity<3,float>::read_from_file(motion_z_filename));
	
  const VoxelsOnCartesianGrid<float> out_density=warp_image(density_sptr, motion_x_sptr, motion_y_sptr, motion_z_sptr, spline_type, extend_borders);
	
  // write image
  Succeeded res = 
    OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
    write_to_file(output_filename, out_density);
  return res==Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}
