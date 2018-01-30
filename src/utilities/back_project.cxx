/*
    Copyright (C) 2014, University College London
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the Lesser GNU General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    Lesser GNU General Public License for more details.

    See STIR/LICENSE.txt for details

*/

/*!
  \file
  \ingroup utilities
  \author Kris Thielemans

  \brief Back project an image.

  \par Usage:
  \verbatim
  back_project output-filename image_to_back_project template_proj_data_file [backprojector-parfile ]\n"
  \endverbatim
  The template_proj_data_file will be used to get the scanner, mashing etc. details
  (its data will \e not be used, nor will it be overwritten).

  The default projector uses the ray-tracing matrix.
  \par Example parameter file for specifying the back projector
  \verbatim
  Back Projector parameters:=
    type := Matrix
      Back projector Using Matrix Parameters :=
        Matrix type := Ray Tracing
         Ray tracing matrix parameters :=
         End Ray tracing matrix parameters :=
        End Back Projector Using Matrix Parameters :=
  End:=
  \endverbatim
*/

#include "stir/ProjData.h"
#include "stir/DiscretisedDensity.h"
#include "stir/IO/read_from_file.h"
#include "stir/recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/IO/OutputFileFormat.h"
#include <iostream>
#include <stdlib.h>

static void print_usage_and_exit()
{
  std::cerr<<"\nUsage:\nback_project output-filename proj_data_to_back_project template_image [backprojector-parfile ]\n";
  std::cerr<<"The default projector uses the ray-tracing matrix.\n\n";
  std::cerr<<"Example parameter file:\n\n"
	   <<"Back Projector parameters:=\n"
	   <<"   type := Matrix\n"
	   <<"   Back projector Using Matrix Parameters :=\n"
	   <<"      Matrix type := Ray Tracing\n"
	   <<"         Ray tracing matrix parameters :=\n"
	   <<"         End Ray tracing matrix parameters :=\n"
	   <<"      End Back Projector Using Matrix Parameters :=\n"
	   <<"End:=\n";

  exit(EXIT_FAILURE);
}


int 
main (int argc, char * argv[])
{
  using namespace stir;

  if (argc!=4 && argc!=5 )
    print_usage_and_exit();
  
  const std::string output_filename = argv[1];

  shared_ptr<ProjData> proj_data_sptr = 
    ProjData::read_from_file(argv[2]);

  shared_ptr <DiscretisedDensity<3,float> > 
    image_density_sptr(read_from_file<DiscretisedDensity<3,float> >(argv[3]));

  shared_ptr<BackProjectorByBin> back_projector_sptr;
  if (argc>=5)
    {
      KeyParser parser;
      parser.add_start_key("Back Projector parameters");
      parser.add_parsing_key("type", &back_projector_sptr);
      parser.add_stop_key("END"); 
      parser.parse(argv[4]);
    }
  else
    {
      shared_ptr<ProjMatrixByBin> PM(new  ProjMatrixByBinUsingRayTracing());
      back_projector_sptr.reset(new BackProjectorByBinUsingProjMatrixByBin(PM)); 
    }

  back_projector_sptr->set_up(proj_data_sptr->get_proj_data_info_ptr()->create_shared_clone(),
			      image_density_sptr );

  image_density_sptr->fill(0.F);
  
  back_projector_sptr->back_project(*image_density_sptr, *proj_data_sptr);
  
  OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
    write_to_file(output_filename, *image_density_sptr);

  return EXIT_SUCCESS;
}

