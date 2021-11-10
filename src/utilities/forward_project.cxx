/*
    Copyright (C) 2014, 2019, 2020 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details

*/

/*!
  \file
  \ingroup utilities
  \author Kris Thielemans

  \brief Forward project an image.

  \par Usage:
  \verbatim
  forward_project output-filename image_to_forward_project template_proj_data_file [forwardprojector-parfile ]\n"
  \endverbatim
  The template_proj_data_file will be used to get the scanner, mashing, ring difference etc. details,
  including energy information (its data will \e not be used, nor will it be overwritten).
  Time frame information is copied from the image.

  Output is currently always in STIR-Interfile format.

  The default projector uses the ray-tracing matrix.
  \par Example parameter file for specifying the forward projector
  \verbatim
  Forward Projector parameters:=
    type := Matrix
      Forward projector Using Matrix Parameters :=
        Matrix type := Ray Tracing
         Ray tracing matrix parameters :=
         End Ray tracing matrix parameters :=
        End Forward Projector Using Matrix Parameters :=
  End:=
  \endverbatim
*/

#include "stir/ProjDataInterfile.h"
#include "stir/DiscretisedDensity.h"
#include "stir/IO/read_from_file.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include <iostream>
#include <stdlib.h>

static void print_usage_and_exit()
{
  std::cerr<<"\nUsage:\nforward_project output-filename image_to_forward_project template_proj_data_file [forwardprojector-parfile ]\n";
  std::cerr<<"The default projector uses the ray-tracing matrix.\n\n";
  std::cerr<<"Example parameter file:\n\n"
	   <<"Forward Projector parameters:=\n"
	   <<"   type := Matrix\n"
	   <<"   Forward projector Using Matrix Parameters :=\n"
	   <<"      Matrix type := Ray Tracing\n"
	   <<"         Ray tracing matrix parameters :=\n"
	   <<"         End Ray tracing matrix parameters :=\n"
	   <<"      End Forward Projector Using Matrix Parameters :=\n"
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

  shared_ptr <DiscretisedDensity<3,float> > 
    image_density_sptr(read_from_file<DiscretisedDensity<3,float> >(argv[2]));

  shared_ptr<ProjData> template_proj_data_sptr = 
    ProjData::read_from_file(argv[3]);

  // create exam_info. Use most things from the image, as often people will
  // just have a standard template
  shared_ptr<ExamInfo> exam_info_sptr(image_density_sptr->get_exam_info().create_shared_clone());

  if (image_density_sptr->get_exam_info().imaging_modality.is_unknown() &&
      template_proj_data_sptr->get_exam_info().imaging_modality.is_known())
    {
      exam_info_sptr->imaging_modality = template_proj_data_sptr->get_exam_info().imaging_modality;
    }
  else if (image_density_sptr->get_exam_info().imaging_modality !=
      template_proj_data_sptr->get_exam_info().imaging_modality)
    error("forward_project: Imaging modality should be the same for the image and the projection data");

  if (template_proj_data_sptr->get_exam_info().has_energy_information())
    {
      if (image_density_sptr->get_exam_info().has_energy_information())
        warning("Both image and template have energy information. Using the latter.");

      exam_info_sptr->set_energy_information_from(template_proj_data_sptr->get_exam_info());
    }

  shared_ptr<ForwardProjectorByBin> forw_projector_sptr;
  if (argc>=5)
    {
      KeyParser parser;
      parser.add_start_key("Forward Projector parameters");
      parser.add_parsing_key("type", &forw_projector_sptr);
      parser.add_stop_key("END"); 
      parser.parse(argv[4]);
    }
  else
    {
      shared_ptr<ProjMatrixByBin> PM(new  ProjMatrixByBinUsingRayTracing());
      forw_projector_sptr.reset(new ForwardProjectorByBinUsingProjMatrixByBin(PM)); 
    }
  if (!forw_projector_sptr)
    {
      std::cerr << "Failure parsing\n";
      return EXIT_FAILURE;
    }

  forw_projector_sptr->set_up(template_proj_data_sptr->get_proj_data_info_sptr()->create_shared_clone(),
                             image_density_sptr );

  ProjDataInterfile output_projdata(exam_info_sptr,
                                    template_proj_data_sptr->get_proj_data_info_sptr()->create_shared_clone(),
                                    output_filename);
  
  forw_projector_sptr->forward_project(output_projdata, *image_density_sptr);
  
  return EXIT_SUCCESS;
}

