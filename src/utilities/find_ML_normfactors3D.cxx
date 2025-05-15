/*
    Copyright (C) 2002, Hammersmith Imanet Ltd
    Copyright (C) 2020, 2022 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

 \file
 \ingroup utilities

 \brief Find normalisation factors using an ML approach

 Just a wrapper around ML_estimate_component_based_normalisation
 \author Kris Thielemans
 */
#include "stir/recon_buildblock/ML_estimate_component_based_normalisation.h"
#include "stir/CPUTimer.h"
#include "stir/info.h"
#include "stir/format.h"
#include "stir/ProjData.h"
#include <iostream>
#include <string>

static void
print_usage_and_exit(const std::string& program_name)
{
  std::cerr << "Usage: " << program_name
            << " [--display | --print-KL | --include-block-timing-model | --for-symmetry-per-block] \\\n"
            << " out_filename_prefix measured_data model num_iterations num_eff_iterations\n"
            << " set num_iterations to 0 to do only efficiencies\n";
  exit(EXIT_FAILURE);
}

USING_NAMESPACE_STIR

int
main(int argc, char** argv)
{
  const char* const program_name = argv[0];
  // skip program name
  --argc;
  ++argv;

  bool do_display = false;
  bool do_KL = false;
  bool do_geo = true;
  bool do_block = false;
  bool do_symmetry_per_block = false;

  // first process command line options
  while (argc > 0 && argv[0][0] == '-' && argc >= 1)
    {
      if (strcmp(argv[0], "--display") == 0)
        {
          do_display = true;
          --argc;
          ++argv;
        }
      else if (strcmp(argv[0], "--print-KL") == 0)
        {
          do_KL = true;
          --argc;
          ++argv;
        }
      else if (strcmp(argv[0], "--include-geometric-model") == 0)
        {
          do_geo = true;
          --argc;
          ++argv;
        }
      else if (strcmp(argv[0], "--include-block-timing-model") == 0)
        {
          do_block = true;
          --argc;
          ++argv;
        }
      else if (strcmp(argv[0], "--for-symmetry-per-block") == 0)
        {
          do_symmetry_per_block = true;
          --argc;
          ++argv;
        }
      else
        print_usage_and_exit(program_name);
    }
  // go back to previous counts such that we don't have to change code below
  ++argc;
  --argv;

  // check_geo_data();
  if (argc != 6)
    {
      print_usage_and_exit(program_name);
    }
  const int num_eff_iterations = atoi(argv[5]);
  const int num_iterations = atoi(argv[4]);
  shared_ptr<ProjData> model_data = ProjData::read_from_file(argv[3]);
  shared_ptr<ProjData> measured_data = ProjData::read_from_file(argv[2]);
  const std::string out_filename_prefix = argv[1];

  CPUTimer timer;
  timer.start();

  ML_estimate_component_based_normalisation(out_filename_prefix,
                                            *measured_data,
                                            *model_data,
                                            num_eff_iterations,
                                            num_iterations,
                                            do_geo,
                                            do_block,
                                            do_symmetry_per_block,
                                            do_KL,
                                            do_display);

  timer.stop();
  info(format("CPU time {} secs", timer.value()));
  return EXIT_SUCCESS;
}
