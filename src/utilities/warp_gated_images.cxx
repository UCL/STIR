//
/*
 Copyright (C) 2009 - 2013, King's College London
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
 
 \brief This program applies motion transformation (warping) and then either accumulates or averages the resulting gated images 
 \author Nicolas A Karakatsanis
 \author Charalampos Tsoumpas
 */
#include "stir/IO/OutputFileFormat.h"
#include "stir/DiscretisedDensity.h"
#include "stir/GatedDiscretisedDensity.h"
#include "stir/spatial_transformation/GatedSpatialTransformation.h"
#include "stir/Succeeded.h"
#include <fstream>
#include <stdio.h>
#include <stdlib.h>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
#endif

USING_NAMESPACE_STIR

using namespace BSpline;

static void print_usage_and_exit()
{
    cerr<<"\nUsage: warp_gated_images <output filename> <filename prefix> <motion vectors prefix> [--accumulation | averaging]\n"
	    <<"\t--accumulation  sums up all the warped gates in the end (default option)\n"
		<<"\t--averaging  takes the average (mean) of all the warped gates in the end\n";
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
  //Nicolas K.: Argument 3rd is now compulsory and cannot be omitted 
  //(not compatible with syntax of older STIR utility: warp_and_accumulate_gated_image)
  if(argc<4 || argc>5)
    print_usage_and_exit();
  
  // initialise this option as default to allow backward-compatibility with 
  // usage of older and still active STIR utility: warp_and_accumulated_gated_images. 
  // Yet 3rd argument here is now compulsory and cannot be omitted
  bool doACCUMULATION=true;
  if (argc==5) 
    {
     if (strcmp(argv[4],"--accumulation")==0)
       doACCUMULATION=true;
     else if (strcmp(argv[4],"--averaging")==0)
       doACCUMULATION=false;
     else
       print_usage_and_exit();
    }

  //	GatedDiscretisedDensity tmp;
  const GatedDiscretisedDensity gated_density(argv[2]);
  GatedSpatialTransformation transformation;
  
  //Nicolas K.: Argument 3rd is now compulsory and cannot be omitted 
  //(not compatible with syntax of older STIR utility: warp_and_accumulate_gated_image)
  transformation.read_from_files(argv[3]);  
  shared_ptr<DiscretisedDensity<3,float> > corrected_image_sptr((gated_density[1]).get_empty_copy());
  
  if (doACCUMULATION)
    transformation.warp_image(*corrected_image_sptr,gated_density);
  else
    transformation.average_warp_image(*corrected_image_sptr,gated_density);
		
  const Succeeded res = OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
    write_to_file(argv[1], *corrected_image_sptr);
  return res==Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}
