/*!

  \file
  \ingroup utilities
  \ingroup GE
  \brief Construct randoms as a product of singles estimates

  Dead-time is not taken into account.

  \todo We currently assume F-18 for decay.

  \author Palak Wadhwa
  \author Kris Thielemans

*/
/*
  Copyright (C) 2001- 2012, Hammersmith Imanet Ltd
  Copyright (C) 2017- 2019, University of Leeds
  Copyright (C) 2020, University College London
  This file is part of STIR.

  This file is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2.0 of the License, or
  (at your option) any later version.

  This file is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  See STIR/LICENSE.txt for details
*/

#include "stir/ProjDataInterfile.h"
#include "stir/decay_correction_factor.h"
#include "stir/ExamInfo.h"
#include "stir/IndexRange2D.h"
#include "stir/IO/GEHDF5Wrapper.h"
#include "stir/info.h"
#include <iostream>
#include <string>
#include "stir/data/SinglesRatesFromGEHDF5.h"
#include "stir/multiply_crystal_factors.h"
#include <boost/format.hpp>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::string;
#endif

USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  if (argc!=4 && argc!=3)
    {
      cerr << "Usage: " << argv[0]
           << " out_filename GE_RDF_filename [template_projdata]\n"
           << "The template is used for size- and time-frame-info, but actual counts are ignored.\n"
           << "If no template is specified, we will use the normal GE sizes and the time frame information of the RDF.\n";
      return EXIT_FAILURE;
    }
  
  const string input_filename = argv[2];
  const string output_file_name = argv[1];
  const string program_name = argv[0];
  shared_ptr<const ProjDataInfo> proj_data_info_sptr;
  shared_ptr<ExamInfo> exam_info_sptr;

  GE::RDF_HDF5::GEHDF5Wrapper input_file(input_filename);
  std::string template_filename;
  if (argc==4)
    {
      template_filename = argv[3];
      shared_ptr<ProjData> template_projdata_sptr = ProjData::read_from_file(template_filename);
      proj_data_info_sptr = template_projdata_sptr->get_proj_data_info_sptr();
      exam_info_sptr = template_projdata_sptr->get_exam_info_sptr();
    }
  else
    {
      template_filename = input_filename;
      proj_data_info_sptr = input_file.get_proj_data_info_sptr();
      exam_info_sptr = input_file.get_exam_info_sptr();
    }

  if (exam_info_sptr->get_time_frame_definitions().get_num_time_frames()==0 ||
      exam_info_sptr->get_time_frame_definitions().get_duration(1) < .0001)
    error("Missing time-frame information in \"" + template_filename +'\"');

  ProjDataInterfile 
    proj_data(exam_info_sptr,
              proj_data_info_sptr->create_shared_clone(),
              output_file_name);

  const int num_rings =
    proj_data_info_sptr->get_scanner_ptr()->get_num_rings();
  const int num_detectors_per_ring =
    proj_data_info_sptr->get_scanner_ptr()->get_num_detectors_per_ring();
  // this uses the wrong naming currently. It so happens that the formulas are the same
  // as when multiplying efficiencies
  Array<2,float> efficiencies(IndexRange2D(num_rings, num_detectors_per_ring));

  {
    GE::RDF_HDF5::SinglesRatesFromGEHDF5  singles;
    singles.read_singles_from_listmode_file(input_filename);
    // efficiencies
    {
      for (int r=0; r<num_rings; ++r)
        for (int c=0; c<num_detectors_per_ring; ++c)
        {
          DetectionPosition<> pos(c,r,0);
          efficiencies[r][c]=singles.get_singles_rate(pos,
                                                      exam_info_sptr->get_time_frame_definitions().get_start_time(1),
                                                      exam_info_sptr->get_time_frame_definitions().get_end_time(1));
        }
    }
  }// nothing

  {
    const float coincidence_time_window = input_file.get_coincidence_time_window();

    /* Randoms from singles formula is

           randoms-rate[i,j] = coinc_window * singles-rate[i] * singles-rate[j]

       However, we actually have total counts in the singles (despite the current name),
       and need total counts in the randoms. This gives

           randoms-counts[i,j] * total_to_activity = coinc_window * singles-counts[i] * singles-counts[j] * total_to_activity^2

       That leads to the formula below.
    */
    const double duration = exam_info_sptr->get_time_frame_definitions().get_duration(1);
    warning("Assuming F-18 tracer!!!");
    const double isotope_halflife = 6586.2;
    const double decay_corr_factor = decay_correction_factor(isotope_halflife, 0., duration);
    const double total_to_activity = decay_corr_factor / duration;
    info(boost::format("decay correction factor: %1%, time frame duration: %2%. total correction factor from activity to counts: %3%")
         % decay_corr_factor % duration % (1/total_to_activity),
         2);

    multiply_crystal_factors(proj_data, efficiencies,
                             coincidence_time_window*total_to_activity);

  }

  return EXIT_SUCCESS;
}
