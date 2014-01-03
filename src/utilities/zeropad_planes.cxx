/*
 Copyright (C) 2011 - 2013, King's College London
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
 
 \brief This program zero pads the start & end planes of an image.
 \author Charalampos Tsoumpas
 */
#include "stir/DiscretisedDensity.h"
#include "stir/Succeeded.h"
#include "stir/IO/OutputFileFormat.h"

USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  if(argc!=4) {
    std::cerr << "Usage: " << argv[0] 
              << " <output filename prefix> <input filename> [number of axial planes] \n";		
    exit(EXIT_FAILURE);
  }
  char const * const output_filename_prefix = argv[1];
  char const * const input_filename = argv[2];
  const int num_planes = atoi(argv[3]);

  const shared_ptr<DiscretisedDensity<3,float> > image_sptr(DiscretisedDensity<3,float>::read_from_file(input_filename));
  const shared_ptr<DiscretisedDensity<3,float> > out_image_sptr(image_sptr->clone());
	
  BasicCoordinate<3,int> c, min, max;
  min[1]=image_sptr->get_min_index();
  max[1]=image_sptr->get_max_index();
	
  for (c[1]=min[1]; c[1]<=min[1]+num_planes-1; ++c[1])
    {
      min[2]=(*image_sptr)[c[1]].get_min_index();
      max[2]=(*image_sptr)[c[1]].get_max_index();
      for (c[2]=min[2]; c[2]<=max[2]; ++c[2])
        {
          min[3]=(*image_sptr)[c[1]][c[2]].get_min_index();
          max[3]=(*image_sptr)[c[1]][c[2]].get_max_index();
          for (c[3]=min[3]; c[3]<=max[3]; ++c[3])
            (*out_image_sptr)[c[1]][c[2]][c[3]]=0.F;
        }
    }
  for (c[1]=max[1]; c[1]>=max[1]-num_planes+1; --c[1])
    {
      min[2]=(*image_sptr)[c[1]].get_min_index();
      max[2]=(*image_sptr)[c[1]].get_max_index();
      for (c[2]=min[2]; c[2]<=max[2]; ++c[2])
        {
          min[3]=(*image_sptr)[c[1]][c[2]].get_min_index();
          max[3]=(*image_sptr)[c[1]][c[2]].get_max_index();
          for (c[3]=min[3]; c[3]<=max[3]; ++c[3])
            (*out_image_sptr)[c[1]][c[2]][c[3]]=0.F;
        }
    }	
  const Succeeded res = OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
    write_to_file(output_filename_prefix, *out_image_sptr);
	
  return res==Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;	
}
