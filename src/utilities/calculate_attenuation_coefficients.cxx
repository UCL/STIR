/*
    Copyright (C) 2001 - 2011-12-31, Hammersmith Imanet Ltd
    Copyright (C) 2013, Kris Thielemans
    Copyright (C) 2015, University College London
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

  \brief Calculates attenuation coefficients (as an alternative to <tt>correct_projdata</tt>).

  \par Usage
  \verbatim
     calculate_attenuation_coefficients
             [--PMRT]  --AF|--ACF <output filename > <input image file name> <template_proj_data>
  \endverbatim
  <tt>--ACF</tt>  calculates the attenuation correction factors, <tt>--AF</tt>  calculates
  the attenuation factor (i.e. the inverse of the ACFs).

  The option <tt>--PMRT</tt> forces forward projection using the Probability Matrix Using Ray Tracing 
  (stir::ProjMatrixByBinUsingRayTracing).

  The attenuation_image has to contain an estimate of the mu-map for the image. It will be used
  to estimate attenuation factors as exp(-forw_proj(*attenuation_image_ptr)).

  \warning attenuation image data are supposed to be in units cm^-1.
  Reference: water has mu .096 cm^-1.

  \author Sanida Mustafovic
  \author Kris Thielemans
*/

#include "stir/ProjDataInterfile.h"
#include "stir/RelatedViewgrams.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/Viewgram.h"
#include "stir/ArrayFunction.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingRayTracing.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/recon_buildblock/BinNormalisationFromAttenuationImage.h"
#include "stir/DataSymmetriesForViewSegmentNumbers.h"
#include "stir/IO/read_from_file.h"
#include "stir/info.h"
#include <boost/format.hpp>
#include <iostream>
#include <list>
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::endl;
using std::cerr;
#endif


START_NAMESPACE_STIR


static void print_usage_and_exit()
{
    std::cerr<<"\nUsage: calculate_attenuation_coefficients [--PMRT]  --AF|--ACF <output filename > <input image file name> <template_proj_data>\n"
	     <<"\t--ACF  calculates the attenuation correction factors\n"
	     <<"\t--AF  calculates the attenuation factor (i.e. the inverse of the ACFs)\n"
             <<"The input image has to give the attenuation (or mu) values at 511 keV, and be in units of cm^-1.\n";
    exit(EXIT_FAILURE);
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int 
main (int argc, char * argv[])
{

  // variable to decide to use the ray-tracing projection matrix or not
  bool use_PMRT=false;

  if (argc>1 && strcmp(argv[1],"--PMRT")==0)
    {
      use_PMRT=true; 
      --argc; ++argv;
    }
  if (argc!=5 )
    print_usage_and_exit();

  bool doACF=true;// initialise to avoid compiler warning
  if (strcmp(argv[1],"--ACF")==0)
    doACF=true;
  else if (strcmp(argv[1],"--AF")==0)
    doACF=false;
  else
    print_usage_and_exit();

  ++argv; --argc;
  
  const std::string atten_image_filename(argv[2]);
  // read it to get ExamInfo
  shared_ptr <DiscretisedDensity<3,float> >
    atten_image_sptr(read_from_file<DiscretisedDensity<3,float> >(atten_image_filename));

  shared_ptr<ProjData> template_proj_data_ptr = 
    ProjData::read_from_file(argv[3]);

  shared_ptr<ForwardProjectorByBin> forw_projector_ptr;
  if (use_PMRT)
    {
      shared_ptr<ProjMatrixByBin> PM(new  ProjMatrixByBinUsingRayTracing());
      forw_projector_ptr.reset(new ForwardProjectorByBinUsingProjMatrixByBin(PM)); 
    }
  else
  {
    forw_projector_ptr.reset(new ForwardProjectorByBinUsingRayTracing());
  }

  cerr << "\n\nForward projector used:\n" << forw_projector_ptr->parameter_info();  

  const std::string output_file_name = argv[1];
  shared_ptr<ProjData> 
    out_proj_data_ptr(
		      new ProjDataInterfile(atten_image_sptr->get_exam_info_sptr(),// TODO this should say it's an ACF File
					    template_proj_data_ptr->get_proj_data_info_ptr()->create_shared_clone(),
					    output_file_name,
                                            std::ios::in|std::ios::out|std::ios::trunc));

  // fill with 1s as we will "normalise" this sinogram.
  out_proj_data_ptr->fill(1.F);

  // construct a normalisation object that does all the work for us.
  shared_ptr<BinNormalisation> normalisation_ptr
	(new BinNormalisationFromAttenuationImage(atten_image_filename,
						  forw_projector_ptr));
  
  if (
      normalisation_ptr->set_up(template_proj_data_ptr->get_proj_data_info_ptr()->create_shared_clone())
      != Succeeded::yes)
    {
      warning("calculate_attenuation_coefficients: set-up of normalisation failed\n");
      return EXIT_FAILURE;
    }

  // dummy values currently necessary for BinNormalisation, but they will be ignored
  const double start_frame = 0;
  const double end_frame = 0;
  shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr(forw_projector_ptr->get_symmetries_used()->clone());
  if (doACF)
    {
      normalisation_ptr->apply(*out_proj_data_ptr,start_frame,end_frame, symmetries_sptr);
    }
  else
    {
      normalisation_ptr->undo(*out_proj_data_ptr,start_frame,end_frame, symmetries_sptr);
    }    

  return EXIT_SUCCESS;
}

