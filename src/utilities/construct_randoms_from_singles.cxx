/*!

  \file
  \ingroup utilities

  \brief Construct randoms as a product of singles estimates

  \author Kris Thielemans

*/
/*
  Copyright (C) 2001- 2012, Hammersmith Imanet Ltd
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
*/

#include "stir/ML_norm.h"

#include "stir/ProjDataInterfile.h"
#include "stir/multiply_crystal_factors.h"
#include "stir/data/randoms_from_singles.h"
#include "stir/decay_correction_factor.h"
#include "stir/stream.h"
#include "stir/IndexRange2D.h"
#include "stir/error.h"
#include <iostream>
#include <fstream>
#include <string>
//#include <algorithm>

using std::cerr;
using std::endl;
using std::ifstream;
using std::string;

USING_NAMESPACE_STIR

int
main(int argc, char** argv)
{
  if (argc != 5)
    {
      cerr << "Usage: " << argv[0] << " out_filename in_norm_filename_prefix template_projdata eff_iter_num\n";
      return EXIT_FAILURE;
    }
  const int eff_iter_num = atoi(argv[4]);
  const int iter_num = 1; // atoi(argv[5]);
  // const bool apply_or_undo = atoi(argv[4])!=0;
  shared_ptr<ProjData> template_projdata_ptr = ProjData::read_from_file(argv[3]);
  const string in_filename_prefix = argv[2];
  const string output_file_name = argv[1];
  const string program_name = argv[0];

  ProjDataInterfile proj_data(template_projdata_ptr->get_exam_info_sptr(),
                              template_projdata_ptr->get_proj_data_info_sptr()->create_shared_clone(),
                              output_file_name);

  const int num_rings = template_projdata_ptr->get_proj_data_info_sptr()->get_scanner_ptr()->get_num_rings();
  const int num_detectors_per_ring
      = template_projdata_ptr->get_proj_data_info_sptr()->get_scanner_ptr()->get_num_detectors_per_ring();
  DetectorEfficiencies efficiencies(IndexRange2D(num_rings, num_detectors_per_ring));

  {

    // efficiencies
    {
      char* in_filename = new char[in_filename_prefix.size() + 30];
      sprintf(in_filename, "%s_%s_%d_%d.out", in_filename_prefix.c_str(), "eff", iter_num, eff_iter_num);
      ifstream in(in_filename);
      in >> efficiencies;
      if (!in)
        {
          error("Error reading %s, using all 1s instead\n", in_filename);
        }
      delete[] in_filename;
    }
  }
  float coincidence_time_window = template_projdata_ptr->get_proj_data_info_sptr()->get_scanner_ptr()->get_coincidence_window_width_in_ps();
  double corr_factor = 1;

  if (coincidence_time_window>0)
  {
      /* Randoms from singles formula is

     randoms-rate[t,i,j] = coinc_window * singles-rate[t,i] * singles-rate[t,j]

     However, we actually have total counts in the singles for sinograms.
     and need total counts in the randoms.
     Assuming there is just decay going on, then we have

     randoms-rate[t,i,j] = coinc_window * singles-rate[0,i] * singles-rate[0,j] exp (-2lambda t)

     randoms-counts[i,j] = int_t1^t2 randoms-rate[t,i,j]
               = coinc_window * singles-rate[0,i] * singles-rate[0,j] * int_t1^t2 exp (-2lambda t)
               = coinc_window * singles-counts[i] * singles-counts[j] *
                 int_t1^t2 exp (-2lambda t) / (int_t1^t2 exp (-lambda t))^2
     where int indicates an integral.

     Now we can use that decay_correction_factor(lambda,t1,t2) computes
        duration/(int_t1^t2 exp (-lambda t))

     That leads to the formula below (as it turns out that the above ratio only depends t2-t1)
  */
      const double isotope_halflife = template_projdata_ptr->get_exam_info().get_radionuclide().get_half_life();
      const TimeFrameDefinitions frame_defs = proj_data.get_exam_info_sptr()->get_time_frame_definitions();
      const double duration = frame_defs.get_duration(1);
      const double decay_corr_factor = decay_correction_factor(isotope_halflife, 0., duration);
      const double double_decay_corr_factor = decay_correction_factor(0.5 * isotope_halflife, 0., duration);
      corr_factor = square(decay_corr_factor) / double_decay_corr_factor / duration;
      info("Correction factor based on coincidence window and duration is "+ std::to_string(corr_factor), 2);

  }

  multiply_crystal_factors(proj_data, efficiencies, static_cast<float>(coincidence_time_window * corr_factor));

  return EXIT_SUCCESS;
}
