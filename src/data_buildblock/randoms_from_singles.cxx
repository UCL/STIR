/*!

  \file
  \ingroup singles_buildblock

  \brief Implementation of stir::randoms_from_singles

  \author Kris Thielemans

*/
/*
  Copyright (C) 2020, 2021, University Copyright London
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0

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
        total_singles[r][c]=singles.get_singles(pos,
                                                frame_defs.get_start_time(1),
                                                frame_defs.get_end_time(1));
      }

  {
    /* Randoms from singles formula is

       randoms-rate[t,i,j] = coinc_window * singles-rate[t,i] * singles-rate[t,j]

       However, we actually have total counts in the singles for sinograms.
       and need total counts in the randoms.
       Assuming there is just decay going on, then we have

       randoms-rate[t,i,j] = coinc_window * singles-rate[0,i] * singles-rate[0,j] exp (-2lambda t)

       randoms-counts[i,j] = int_t1^t2 randoms-rate[t,i,j]
                 = coinc_window * singles-rate[0,i] * singles-rate[0,j] * int_t1^t2 exp (-2lambda t)
                   coinc_window * singles-counts[i] * singles-counts[j] *
                   int_t1^t2 exp (-2lambda t) / (int_t1^t2 exp (-lambda t))^2

       Now we can use that decay_correction_factor(lambda,t1,t2) computes
          duration/(int_t1^t2 exp (-lambda t))

       That leads to the formula below (as it turns out that the above ratio only depends t2-t1)
    */
    const double duration = frame_defs.get_duration(1);
    warning("Assuming F-18 tracer!!!");
    const double isotope_halflife = 6586.2;
    const double decay_corr_factor = decay_correction_factor(isotope_halflife, 0., duration);
    const double double_decay_corr_factor = decay_correction_factor(2*isotope_halflife, 0., duration);
    const double corr_factor = square(decay_corr_factor) / double_decay_corr_factor / duration;
    info(boost::format("RFS: decay correction factor: %1%, time frame duration: %2%. total correction factor from (singles_totals)^2 to randoms_totals: %3%")
         % decay_corr_factor % duration % (1/corr_factor),
         2);

    multiply_crystal_factors(proj_data, total_singles,
                             static_cast<float>(coincidence_time_window*corr_factor));

  }
}

END_NAMESPACE_STIR
