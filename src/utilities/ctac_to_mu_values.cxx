//
//
/*
  Copyright (C) 2019, 2020, Institute of Nuclear Medicine, University College London
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup utilities
  \brief Produces an image of mu-values from a CT/CTAC.
  \author Benjamin A. Thomas
  \author Kris Thielemans

  \par Usage:
  \code
  ctac_to_mu_values -o output_filename -i input_volume -j slope_filename -m manufacturer_name [-v kilovoltage_peak] -k
  target_photon_energy \endcode Default value for tube voltage is 120 kV. For PET, the \c target_photon_energy has to be 511.

  This convert HU to mu-values using a piece-wise linear curve. \see HUToMuImageProcessor for details on file format etc

  \warning This utility currently does not implement post-filtering to PET resolution, nor resampling to the PET voxel-size.

*/

#include "stir/info.h"
#include "stir/Succeeded.h"
#include "stir/HUToMuImageProcessor.h"
#include "stir/IO/read_from_file.h"
#include "stir/IO/write_to_file.h"
#include "stir/DiscretisedDensity.h"
#include "stir/getopt.h"
#include "stir/warning.h"
#include "stir/format.h"
#include <string>
#include <exception>

USING_NAMESPACE_STIR

typedef DiscretisedDensity<3, float> FloatImageType;

int
main(int argc, char* argv[])
{
  USING_NAMESPACE_STIR;
  const char* output_filename = 0;
  const char* input_filename = 0;
  const char* slope_filename = 0;
  const char* manufacturer_name = 0;
  const char* kVp_str = 0;
  const char* keV_str = 0;

  const char* const usage = "ctac_to_mu_values -o output_filename -i input_volume -j slope_filename -m manufacturer_name [-v "
                            "kilovoltage_peak] -k target_photon_energy\n"
                            "Default value for tube voltage is 120 kV.\n";
  opterr = 0;
  {
    char c;

    while ((c = getopt(argc, argv, "i:o:j:m:v:k:?")) != -1)
      switch (c)
        {
        case 'i':
          input_filename = optarg;
          break;
        case 'o':
          output_filename = optarg;
          break;
        case 'j':
          slope_filename = optarg;
          break;
        case 'm':
          manufacturer_name = optarg;
          break;
        case 'v':
          kVp_str = optarg;
          break;
        case 'k':
          keV_str = optarg;
          break;
        case '?':
          std::cerr << usage;
          return EXIT_FAILURE;
        default:
          if (isprint(optopt))
            fprintf(stderr, "Unknown option `-%c'.\n", optopt);
          else
            fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
          std::cerr << usage;
          return EXIT_FAILURE;
        }
  }

  if (output_filename == 0 || input_filename == 0 || slope_filename == 0 || manufacturer_name == 0 || keV_str == 0)
    {
      std::cerr << usage;
      return EXIT_FAILURE;
    }

  if (kVp_str == 0)
    {
      // If the user does not specify a value for kVP, assume 120 kVp.
      kVp_str = "120";
      stir::info(format("No value for kVp given, assuming {}", kVp_str));
    }

  try
    {
      HUToMuImageProcessor<DiscretisedDensity<3, float>> hu_to_mu;
      hu_to_mu.set_manufacturer_name(manufacturer_name);
      hu_to_mu.set_slope_filename(slope_filename);
      hu_to_mu.set_target_photon_energy(std::stof(keV_str));
      hu_to_mu.set_kilovoltage_peak(std::stof(kVp_str));
      // Read DICOM data
      stir::info(format("ctac_to_mu_values: opening file {}", input_filename));
      unique_ptr<FloatImageType> input_image_sptr(stir::read_from_file<FloatImageType>(input_filename));
      hu_to_mu.set_up(*input_image_sptr);

      // Create output image from input image.
      shared_ptr<FloatImageType> output_image_sptr(input_image_sptr->clone());
      // Apply scaling.
      hu_to_mu.apply(*output_image_sptr, *input_image_sptr);
      // Write output file.
      write_to_file(output_filename, *output_image_sptr);
    }
  catch (std::string& error_string)
    {
      // don't print yet, as error() already does that at the moment
      // std::cerr << error_string << std::endl;
      return EXIT_FAILURE;
    }
  catch (std::exception& e)
    {
      stir::warning(e.what());
      return EXIT_FAILURE;
    }
  catch (...)
    {
      return EXIT_FAILURE;
    }
  return EXIT_SUCCESS;
}
