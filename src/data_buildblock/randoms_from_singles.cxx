/*!

  \file
  \ingroup singles_buildblock

  \brief Implementation of stir::randoms_from_singles

  \author Kris Thielemans

*/
/*
  Copyright (C) 2020, 2021, University Copyright London
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

#include "stir/multiply_crystal_factors.h"
#include "stir/ProjData.h"
#include "stir/data/SinglesRates.h"
#include "stir/Scanner.h"
#include "stir/DetectionPosition.h"
#include "stir/TimeFrameDefinitions.h"
#include "stir/decay_correction_factor.h"
#include "stir/IndexRange2D.h"
#include "stir/info.h"

START_NAMESPACE_STIR

void randoms_from_singles(ProjData& proj_data, const SinglesRates& singles,
                          const float coincidence_time_window)
{
  const int num_rings =
    proj_data.get_proj_data_info_sptr()->get_scanner_ptr()->get_num_rings();
  const int num_detectors_per_ring =
    proj_data.get_proj_data_info_sptr()->get_scanner_ptr()->get_num_detectors_per_ring();

  const TimeFrameDefinitions frame_defs = proj_data.get_exam_info_sptr()->get_time_frame_definitions();

  // get total singles for this frame
  Array<2,float> total_singles(IndexRange2D(num_rings, num_detectors_per_ring));
  for (int r=0; r<num_rings; ++r)
    for (int c=0; c<num_detectors_per_ring; ++c)
      {
        const DetectionPosition<> pos(c,r,0);
        total_singles[r][c]=singles.get_singles_rate(pos,
                                                     frame_defs.get_start_time(1),
                                                     frame_defs.get_end_time(1));
      }

  {
    /* Randoms from singles formula is

       randoms-rate[i,j] = coinc_window * singles-rate[i] * singles-rate[j]

       However, we actually have total counts in the singles (despite the current name),
       and need total counts in the randoms. This gives

       randoms-counts[i,j] * total_to_activity = coinc_window * singles-counts[i] * singles-counts[j] * total_to_activity^2

       That leads to the formula below.
    */
    const double duration = frame_defs.get_duration(1);
    warning("Assuming F-18 tracer!!!");
    const double isotope_halflife = 6586.2;
    const double decay_corr_factor = decay_correction_factor(isotope_halflife, 0., duration);
    const double total_to_activity = decay_corr_factor / duration;
    info(boost::format("RFS: decay correction factor: %1%, time frame duration: %2%. total correction factor from activity to counts: %3%")
         % decay_corr_factor % duration % (1/total_to_activity),
         2);

    multiply_crystal_factors(proj_data, total_singles,
                             static_cast<float>(coincidence_time_window*total_to_activity));

  }
}

END_NAMESPACE_STIR
