/*!

  \file
  \ingroup projdata

  \brief Implementation of stir::multiply_crystal_factors

  \author Kris Thielemans

*/
/*
  Copyright (C) 2021, 2022, 2024 University Copyright London
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
*/

#include "stir/multiply_crystal_factors.h"
#include "stir/ProjData.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjDataInfoBlocksOnCylindricalNoArcCorr.h"
#include "stir/Bin.h"
#include "stir/Sinogram.h"
#include "stir/error.h"
#include <memory>

START_NAMESPACE_STIR

// declaration of local function that does the work
template <class TProjDataInfo>
void
multiply_crystal_factors_help(ProjData& proj_data,
                              const TProjDataInfo& proj_data_info,
                              const Array<2, float>& efficiencies,
                              float global_factor)
{
  // we will duplicate TOF sinograms, so need to divide with their number such that
  // total remains preserved
  global_factor /= proj_data.get_num_tof_poss();

  const auto non_tof_proj_data_info_sptr = std::dynamic_pointer_cast<TProjDataInfo>(proj_data_info.create_non_tof_clone());
  Bin bin;

  for (bin.segment_num() = proj_data.get_min_segment_num(); bin.segment_num() <= proj_data.get_max_segment_num();
       ++bin.segment_num())
    {

      for (bin.axial_pos_num() = proj_data.get_min_axial_pos_num(bin.segment_num());
           bin.axial_pos_num() <= proj_data.get_max_axial_pos_num(bin.segment_num());
           ++bin.axial_pos_num())
        {
          Sinogram<float> sinogram = non_tof_proj_data_info_sptr->get_empty_sinogram(SinogramIndices(bin));

#ifdef STIR_OPENMP
#  if _OPENMP >= 200711
#    pragma omp parallel for collapse(2) // OpenMP 3.1
#  else
#    pragma omp parallel for // older versions
#  endif
#endif
          for (int view_num = proj_data.get_min_view_num(); view_num <= proj_data.get_max_view_num(); ++view_num)
            {
              for (int tangential_pos_num = proj_data.get_min_tangential_pos_num();
                   tangential_pos_num <= proj_data.get_max_tangential_pos_num();
                   ++tangential_pos_num)
                {
                  // Construct bin with appropriate values
                  // Sadly cannot be done in the loops above for OpenMP 2.0 compatibility
                  Bin parallel_bin(bin);
                  parallel_bin.view_num() = view_num;
                  parallel_bin.tangential_pos_num() = tangential_pos_num;

                  std::vector<DetectionPositionPair<>> det_pos_pairs;
                  non_tof_proj_data_info_sptr->get_all_det_pos_pairs_for_bin(
                      det_pos_pairs, parallel_bin); // using the default argument to ignore TOF here
                  float result = 0.F;
                  for (unsigned int i = 0; i < det_pos_pairs.size(); ++i)
                    {
                      const auto& p1 = det_pos_pairs[i].pos1();
                      const auto& p2 = det_pos_pairs[i].pos2();
                      result += efficiencies[p1.axial_coord()][p1.tangential_coord()]
                                * efficiencies[p2.axial_coord()][p2.tangential_coord()];
                    }
#if defined(STIR_OPENMP)
#  if _OPENMP >= 201012
#    pragma omp atomic update
#  else
#    pragma omp critical(STIRMULTIPLYCRYSTALFACTORS)
                  {
#  endif
#endif
                  // Use += such that the "atomic update" pragma compiles (OpenMP 3.0).
                  // Presumably with OpenMP 3.1 we could use "atomic write"
                  sinogram[parallel_bin.view_num()][parallel_bin.tangential_pos_num()] += result * global_factor;
#if defined(STIR_OPENMP) && _OPENMP < 201012
                }
#endif
            }
        }
      // now set sinogram, a bit complicated for TOF as we replicate
      if (proj_data.get_num_tof_poss() == 1)
        {
          proj_data.set_sinogram(sinogram);
        }
      else
        {
          for (bin.timing_pos_num() = proj_data.get_min_tof_pos_num(); bin.timing_pos_num() <= proj_data.get_max_tof_pos_num();
               ++bin.timing_pos_num())
            {
              // construct TOF sinogram with same values as the non-TOF sinogram,
              // but appropriate meta-data.
              const Sinogram<float> tof_sinogram(sinogram, proj_data.get_proj_data_info_sptr(), SinogramIndices(bin));
              proj_data.set_sinogram(tof_sinogram);
            }
        }
    }
}
}

void
multiply_crystal_factors(ProjData& proj_data, const Array<2, float>& efficiencies, const float global_factor)
{
  if (proj_data.get_proj_data_info_sptr()->get_scanner_ptr()->get_scanner_geometry() == "Cylindrical")
    {
      auto proj_data_info_ptr
          = dynamic_cast<const ProjDataInfoCylindricalNoArcCorr* const>(proj_data.get_proj_data_info_sptr().get());

      if (proj_data_info_ptr == 0)
        {
          error("Can only process not arc-corrected data\n");
        }
      multiply_crystal_factors_help(proj_data, *proj_data_info_ptr, efficiencies, global_factor);
    }
  else
    {
      auto proj_data_info_ptr
          = dynamic_cast<const ProjDataInfoBlocksOnCylindricalNoArcCorr* const>(proj_data.get_proj_data_info_sptr().get());

      if (proj_data_info_ptr == 0)
        {
          error("Can only process not arc-corrected data\n");
        }
      multiply_crystal_factors_help(proj_data, *proj_data_info_ptr, efficiencies, global_factor);
    }
}

template void
multiply_crystal_factors_help(ProjData&, const ProjDataInfoCylindricalNoArcCorr&, const Array<2, float>&, const float);
template void
multiply_crystal_factors_help(ProjData&, const ProjDataInfoBlocksOnCylindricalNoArcCorr&, const Array<2, float>&, const float);
END_NAMESPACE_STIR
