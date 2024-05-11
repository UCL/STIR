/*
    Copyright (C) 2001- 2012, Hammersmith Imanet Ltd
    Copyright (C) 2020, 2022, University College London
    Copyright (C) 2016-2017, PETsys Electronics
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup utilities

  \brief Apply normalisation factors estimated using the ML code to projection data

  This utility constructed a stir::BinNormalisationPETFromComponents from the text
  files written by stir::ML_estimate_component_based_normalisation and uses it
  on projection data.
  \author Kris Thielemans
  \author Tahereh Niknejad

*/

#include "stir/recon_buildblock/BinNormalisationPETFromComponents.h"

#include "stir/ProjDataInterfile.h"
#include "stir/ProjDataInMemory.h"
#include "stir/stream.h"
#include "stir/warning.h"
#include <iostream>
#include <string>

USING_NAMESPACE_STIR

int
main(int argc, char** argv)
{
  if (argc < 7 || argc > 13)
    {
      std::cerr << "Usage: " << argv[0]
                << " out_filename in_norm_filename_prefix measured_data multiply_or_divide iter_num eff_iter_num\\\n"
                << "\t [do_eff [ do_geo [do_block [do_display [do_symmetry_per_block ]]]]]\n"
                << "multiply_or_divide is 1 (multiply) or 0 (divide), with the latter being \"correction\"\n"
                << "do_eff, do_geo, do_block are 1 or 0 and all default to 1\n"
                << "do_display is ignored\n"
                << "do_symmetry_per_block is 1 or 0 (defaults to 0)\n";
      return EXIT_FAILURE;
    }

  bool do_symmetry_per_block = argc >= 12 ? atoi(argv[11]) != 0 : false;
  // const bool do_display = argc >= 11 ? atoi(argv[10]) != 0 : false;
  bool do_block = argc >= 10 ? atoi(argv[9]) != 0 : true;
  bool do_geo = argc >= 9 ? atoi(argv[8]) != 0 : true;
  bool do_eff = argc >= 8 ? atoi(argv[7]) != 0 : true;

  const int eff_iter_num = atoi(argv[6]);
  const int iter_num = atoi(argv[5]);
  const bool multiply_or_divide = atoi(argv[4]) != 0;
  shared_ptr<ProjData> measured_data = ProjData::read_from_file(argv[3]);
  const std::string in_filename_prefix = argv[2];
  const std::string output_file_name = argv[1];
  // const std::string program_name = argv[0];
  ProjDataInterfile out_proj_data(
      measured_data->get_exam_info_sptr(), measured_data->get_proj_data_info_sptr(), output_file_name);

  BinNormalisationPETFromComponents norm;
  norm.allocate(measured_data->get_proj_data_info_sptr(), do_eff, do_geo, do_block, do_symmetry_per_block);
  {

    // efficiencies
    if (do_eff)
      {
        char* in_filename = new char[in_filename_prefix.size() + 30];
        sprintf(in_filename, "%s_%s_%d_%d.out", in_filename_prefix.c_str(), "eff", iter_num, eff_iter_num);
        std::ifstream in(in_filename);
        in >> norm.crystal_efficiencies();
        if (!in)
          {
            warning("Error reading %s, using all 1s instead\n", in_filename);
            do_eff = false;
          }
        delete[] in_filename;
      }

    // geo norm
    if (do_geo)
      {
        {
          char* in_filename = new char[in_filename_prefix.size() + 30];
          sprintf(in_filename, "%s_%s_%d.out", in_filename_prefix.c_str(), "geo", iter_num);
          std::ifstream in(in_filename);
          in >> norm.geometric_factors();
          if (!in)
            {
              warning("Error reading %s, using all 1s instead\n", in_filename);
              do_geo = false;
            }
          delete[] in_filename;
        }
      }

    // block norm
    if (do_block)
      {
        {
          char* in_filename = new char[in_filename_prefix.size() + 30];
          sprintf(in_filename, "%s_%s_%d.out", in_filename_prefix.c_str(), "block", iter_num);
          std::ifstream in(in_filename);
          in >> norm.block_factors();
          if (!in)
            {
              warning("Error reading %s, using all 1s instead\n", in_filename);
              do_block = false;
            }
          delete[] in_filename;
        }
      }
  }

  norm.set_up(measured_data->get_exam_info_sptr(), measured_data->get_proj_data_info_sptr());
  ProjDataInMemory proj_data(*measured_data);
  if (multiply_or_divide)
    norm.undo(proj_data);
  else
    norm.apply(proj_data);
  out_proj_data.fill(proj_data);

  return EXIT_SUCCESS;
}
