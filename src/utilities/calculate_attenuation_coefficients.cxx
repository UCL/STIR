/*
    Copyright (C) 2001 - 2011-12-31, Hammersmith Imanet Ltd
    Copyright (C) 2013, Kris Thielemans
    Copyright (C) 2015, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

/*!
  \file
  \ingroup utilities

  \brief Calculates attenuation coefficients (as an alternative to <tt>correct_projdata</tt>).

  \par Usage
  \verbatim
     calculate_attenuation_coefficients
             [--PMRT --NOPMRT]  --AF|--ACF <output filename > <input image file name> <template_proj_data>
  [forwardprojector-parfile] \endverbatim <tt>--ACF</tt>  calculates the attenuation correction factors, <tt>--AF</tt>  calculates
  the attenuation factor (i.e. the inverse of the ACFs).

  The option <tt>--PMRT</tt> forces forward projection using the Probability Matrix Using Ray Tracing
  (stir::ProjMatrixByBinUsingRayTracing).

  The option <tt>--NOPMRT</tt> forces forward projection using the (old) Ray Tracing

  \par Optionally include a parameter file for specifying the forward projector (overrules --PMRT and --NOPMRT options)
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

  The attenuation_image has to contain an estimate of the mu-map for the image. It will be used
  to estimate attenuation factors as exp(-forw_proj(*attenuation_image_ptr)).

  \par Note

  Output is non-TOF, even if a TOF template is used.

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
#include "stir/warning.h"
#include <boost/format.hpp>
#include <iostream>
#include <list>
#include <algorithm>

using std::endl;
using std::cerr;

START_NAMESPACE_STIR

static void
print_usage_and_exit()
{
  std::cerr << "\nUsage: calculate_attenuation_coefficients [--PMRT --NOPMRT]  --AF|--ACF <output filename > <input image file "
               "name> <template_proj_data> [forwardprojector-parfile]\n"
            << "\t--ACF  calculates the attenuation correction factors\n"
            << "\t--AF  calculates the attenuation factor (i.e. the inverse of the ACFs)\n"
            << "\t--PMRT uses the Ray Tracing Projection Matrix (default) (ignored if parfile provided)\n"
            << "\t--NOPMRT uses the (old) Ray Tracing forward projector (ignored if parfile provided)\n"
            << "The input image has to give the attenuation (or mu) values at 511 keV, and be in units of cm^-1.\n\n"
            << "Example forward projector parameter file:\n\n"
            << "Forward Projector parameters:=\n"
            << "   type := Matrix\n"
            << "   Forward projector Using Matrix Parameters :=\n"
            << "      Matrix type := Ray Tracing\n"
            << "         Ray tracing matrix parameters :=\n"
            << "         End Ray tracing matrix parameters :=\n"
            << "      End Forward Projector Using Matrix Parameters :=\n"
            << "End:=\n";

  exit(EXIT_FAILURE);
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int
main(int argc, char* argv[])
{

  // variable to decide to use the ray-tracing projection matrix or not
  bool use_PMRT = true;

  if (argc > 1 && strcmp(argv[1], "--PMRT") == 0)
    {
      use_PMRT = true;
      --argc;
      ++argv;
    }
  if (!(argc == 5 || argc == 6))
    print_usage_and_exit();

  bool doACF = true; // initialise to avoid compiler warning
  if (strcmp(argv[1], "--ACF") == 0)
    doACF = true;
  else if (strcmp(argv[1], "--AF") == 0)
    doACF = false;
  else
    print_usage_and_exit();

  ++argv;
  --argc;

  const std::string atten_image_filename(argv[2]);
  // read it to get ExamInfo
  shared_ptr<DiscretisedDensity<3, float>> atten_image_sptr(read_from_file<DiscretisedDensity<3, float>>(atten_image_filename));

  shared_ptr<ProjData> template_proj_data_ptr = ProjData::read_from_file(argv[3]);

  shared_ptr<ForwardProjectorByBin> forw_projector_sptr;
  if (argc >= 5)
    {
      KeyParser parser;
      parser.add_start_key("Forward Projector parameters");
      parser.add_parsing_key("type", &forw_projector_sptr);
      parser.add_stop_key("END");
      parser.parse(argv[4]);
    }
  else if (use_PMRT)
    {
      shared_ptr<ProjMatrixByBin> PM(new ProjMatrixByBinUsingRayTracing());
      forw_projector_sptr.reset(new ForwardProjectorByBinUsingProjMatrixByBin(PM));
    }
  else
    {
      forw_projector_sptr.reset(new ForwardProjectorByBinUsingRayTracing());
    }

  cerr << "\n\nForward projector used:\n" << forw_projector_sptr->parameter_info();

  if (template_proj_data_ptr->get_proj_data_info_sptr()->is_tof_data())
    {
      info("The scanner template provided contains TOF information. The calculation of the attenuation coefficients will be "
           "non-TOF anyway.");
    }

  const std::string output_file_name = argv[1];
  shared_ptr<ProjData> out_proj_data_ptr(
      new ProjDataInterfile(template_proj_data_ptr->get_exam_info_sptr(), // TODO this should say it's an ACF File
                            template_proj_data_ptr->get_proj_data_info_sptr()->create_non_tof_clone(),
                            output_file_name,
                            std::ios::in | std::ios::out | std::ios::trunc));

  // fill with 1s as we will "normalise" this sinogram.
  out_proj_data_ptr->fill(1.F);

  // construct a normalisation object that does all the work for us.
  shared_ptr<BinNormalisation> normalisation_ptr(
      new BinNormalisationFromAttenuationImage(atten_image_filename, forw_projector_sptr));

  if (normalisation_ptr->set_up(template_proj_data_ptr->get_exam_info_sptr(),
                                template_proj_data_ptr->get_proj_data_info_sptr()->create_non_tof_clone())
      != Succeeded::yes)
    {
      warning("calculate_attenuation_coefficients: set-up of normalisation failed\n");
      return EXIT_FAILURE;
    }

  shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr(forw_projector_sptr->get_symmetries_used()->clone());
  if (doACF)
    {
      normalisation_ptr->apply(*out_proj_data_ptr, symmetries_sptr);
    }
  else
    {
      normalisation_ptr->undo(*out_proj_data_ptr, symmetries_sptr);
    }

  return EXIT_SUCCESS;
}
