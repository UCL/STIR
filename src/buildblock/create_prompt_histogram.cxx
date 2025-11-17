/*!

  \file
  \ingroup projdata

  \brief Implementation of stir::create_prompt_histogram

  \author Kris Thielemans

*/
/*
  Copyright (C) 2021, 2022, 2024 University Copyright London
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
*/

#include "stir/create_prompt_histogram.h"
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
create_prompt_histogram_help(Array<2, float>& prompt_histogram, const ProjData& prompt_proj_data, const TProjDataInfo& proj_data_info)
{
  const auto non_tof_proj_data_info_sptr = std::dynamic_pointer_cast<TProjDataInfo>(proj_data_info.create_non_tof_clone());
  Bin bin;

  for (bin.segment_num() = prompt_proj_data.get_min_segment_num(); bin.segment_num() <= prompt_proj_data.get_max_segment_num();
       ++bin.segment_num())
    {
      for (bin.axial_pos_num() = prompt_proj_data.get_min_axial_pos_num(bin.segment_num());
           bin.axial_pos_num() <= prompt_proj_data.get_max_axial_pos_num(bin.segment_num());
           ++bin.axial_pos_num())
        {
          auto sinogram = prompt_proj_data.get_sinogram(bin.axial_pos_num(), bin.segment_num());
#ifdef STIR_OPENMP
#  if _OPENMP >= 200711
#    pragma omp parallel for collapse(2) // OpenMP 3.1
#  else
#    pragma omp parallel for // older versions
#  endif
#endif
          for (int view_num = prompt_proj_data.get_min_view_num(); view_num <= prompt_proj_data.get_max_view_num(); ++view_num)
            {
              for (int tangential_pos_num = prompt_proj_data.get_min_tangential_pos_num();
                   tangential_pos_num <= prompt_proj_data.get_max_tangential_pos_num();
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
                  for (unsigned int i = 0; i < det_pos_pairs.size(); ++i)
                    {
                      const auto& p1 = det_pos_pairs[i].pos1();
                      const auto& p2 = det_pos_pairs[i].pos2();
                      const auto count = sinogram[parallel_bin.view_num()][parallel_bin.tangential_pos_num()];
#if defined(STIR_OPENMP)
#  if _OPENMP >= 201012
#    pragma omp atomic update
#  else
#    pragma omp critical(STIRCREATEPROMPTHISTOGRAM)
                      {
#  endif
#endif
                      prompt_histogram[p1.axial_coord()][p1.tangential_coord()] += count;
#if defined(STIR_OPENMP)
#  if _OPENMP >= 201012
#    pragma omp atomic update
#  endif
#endif
                      prompt_histogram[p2.axial_coord()][p2.tangential_coord()] += count;
#if defined(STIR_OPENMP) && _OPENMP < 201012
                      }
#endif
                    }
                }
              
            }
        }
    }
}

void
create_prompt_histogram(Array<2, float>& prompt_histogram, const ProjData& prompt_proj_data)
{
  if (prompt_proj_data.get_proj_data_info_sptr()->get_scanner_ptr()->get_scanner_geometry() == "Cylindrical")
    {
      auto proj_data_info_ptr
          = dynamic_cast<const ProjDataInfoCylindricalNoArcCorr* const>(prompt_proj_data.get_proj_data_info_sptr().get());

      if (proj_data_info_ptr == 0)
        {
          error("Can only process not arc-corrected data\n");
        }
      create_prompt_histogram_help(prompt_histogram, prompt_proj_data, *proj_data_info_ptr);
    }
  else
    {
      auto proj_data_info_ptr
          = dynamic_cast<const ProjDataInfoBlocksOnCylindricalNoArcCorr* const>(prompt_proj_data.get_proj_data_info_sptr().get());

      if (proj_data_info_ptr == 0)
        {
          error("Can only process not arc-corrected data\n");
        }
      create_prompt_histogram_help(prompt_histogram, prompt_proj_data, *proj_data_info_ptr);
    }
}

template void
create_prompt_histogram_help(Array<2, float>&, const ProjData&, const ProjDataInfoCylindricalNoArcCorr&);
template void
create_prompt_histogram_help(Array<2, float>&, const ProjData&, const ProjDataInfoBlocksOnCylindricalNoArcCorr&);

END_NAMESPACE_STIR
