//
//
/*!
  \file
  \ingroup Parallelproj

  \brief non-inline implementations for stir::ParallelprojHelper

  \author Kris Thielemans
  \author Nicole Jurjew  
*/
/*
    Copyright (C) 2021, 2023, 2024 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/recon_buildblock/Parallelproj_projector/ParallelprojHelper.h"
#include "stir/ProjData.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/LORCoordinates.h"
#include "stir/Bin.h"
#include "stir/TOF_conversions.h"

// for debugging, remove later
#include "stir/info.h"
#include "stir/stream.h"
#include <iostream>
#include "stir/num_threads.h"

START_NAMESPACE_STIR

detail::ParallelprojHelper::~ParallelprojHelper()
{
}

template <class T>
static inline void copy_to_array(const BasicCoordinate<3,T>& c, std::array<T,3>& a)
{
  std::copy(c.begin(), c.end(), a.begin());
}

detail::ParallelprojHelper::ParallelprojHelper(const ProjDataInfo& p_info, const DiscretisedDensity<3,float> &density) :
  xstart(p_info.size_all()*3),
  xend(p_info.size_all()*3)
{
  info("Creating parallelproj data-structures", 2);

  auto& stir_image = dynamic_cast<const VoxelsOnCartesianGrid<float>&>(density);
  
  auto stir_voxel_size = stir_image.get_voxel_size();
#ifndef NEWSCALE
  // parallelproj projectors work in units of the voxel_size passed.
  // STIR projectors have to be in pixel units, so convert the voxel-size
  const float rescale = 1/stir_voxel_size[3];
#else
  const float rescale = 1.F;
#endif

    num_image_voxel = static_cast<long long>(stir_image.size_all());
    num_lors = static_cast<long long>(p_info.size_all())/p_info.get_num_tof_poss();

    sigma_tof = tof_delta_time_to_mm(p_info.get_scanner_sptr()->get_timing_resolution())/2.355*rescale;
    tofcenter_offset = 0.F*rescale;
    Bin bin(0,0,0,0,0);
    tofbin_width = p_info.get_sampling_in_k(bin)*rescale;
    num_tof_bins = p_info.get_num_tof_poss();

  copy_to_array(stir_voxel_size*rescale, voxsize);
  copy_to_array(stir_image.get_lengths(), imgdim);

  BasicCoordinate<3,int> min_indices, max_indices;
  stir_image.get_regular_range(min_indices, max_indices);
  auto coord_first_voxel = stir_image.get_physical_coordinates_for_indices(min_indices);
  // TODO origin shift get_LOR() uses the "centred w.r.t. gantry" coordinate system
  coord_first_voxel[1] -= (stir_image.get_min_index() + stir_image.get_max_index())/2.F * stir_voxel_size[1];
  copy_to_array(coord_first_voxel*rescale, origin);

  // loop over all LORs in the projdata
  const float radius = p_info.get_scanner_sptr()->get_max_FOV_radius();

  // warning: next loop needs to be the same as how ProjDataInMemory stores its data. There is no guarantee that this will remain the case in the future.
  const auto segment_sequence = ProjData::standard_segment_sequence(p_info);
  std::size_t index(0);

#ifdef STIR_OPENMP
  // Using too many threads is counterproductive according to my timings, so I limited to 8 (not necessarily optimal!).
  const auto num_threads_to_use = std::min(8,get_max_num_threads());
#  if _OPENMP >=201012
#    define ATOMICWRITE _Pragma("omp atomic write") \

#    define CRITICALSECTION
#  else
#    define ATOMICWRITE
#    if defined(_MSC_VER) && (_MSC_VER < 1910)
       // no _Pragma until VS 2017
#      define CRITICALSECTION
#    else
#      define CRITICALSECTION _Pragma("omp critical(PARALLELPROJHELPER_INIT)")
#    endif
#  endif
#else
#  define ATOMICWRITE
#  define CRITICALSECTION
#endif
  for (int seg : segment_sequence)
    {
      for (int axial_pos_num = p_info.get_min_axial_pos_num(seg); axial_pos_num <= p_info.get_max_axial_pos_num(seg); ++axial_pos_num)
        {
          for (int view_num = p_info.get_min_view_num(); view_num <= p_info.get_max_view_num(); ++view_num)
            {
#ifdef STIR_OPENMP
              #pragma omp parallel for num_threads(num_threads_to_use)
#endif
              for (int tangential_pos_num = p_info.get_min_tangential_pos_num(); tangential_pos_num <= p_info.get_max_tangential_pos_num(); ++tangential_pos_num)
                {
                  Bin bin;
                  bin.segment_num() = seg;
                  bin.axial_pos_num() = axial_pos_num;
                  bin.view_num() = view_num;
                  bin.tangential_pos_num() = tangential_pos_num;
                  // compute index for this bin (independent of multi-threading)
                  const std::size_t this_index = index + (bin.tangential_pos_num() - p_info.get_min_tangential_pos_num())*3;
                  LORInAxialAndNoArcCorrSinogramCoordinates<float> lor;
                  LORAs2Points<float> lor_points;

                  p_info.get_LOR(lor, bin);
                  if (lor.get_intersections_with_cylinder(lor_points, radius) == Succeeded::no)
                    {
                      // memory is already allocated, so just passing in points that will produce nothing
                      CRITICALSECTION
                        {
                          ATOMICWRITE xstart[this_index] = 0;
                          ATOMICWRITE xend[this_index] = 0;
                          ATOMICWRITE xstart[this_index+1] = 0;
                          ATOMICWRITE xend[this_index+1] = 0;
                          ATOMICWRITE xstart[this_index+2] = 0;
                          ATOMICWRITE xend[this_index+2] = 0;
                        }
                  }
                  else
                  {
                    const auto p1 = lor_points.p1()*rescale;
                    const auto p2 = lor_points.p2()*rescale;
                    CRITICALSECTION
                      {
                        ATOMICWRITE xstart[this_index] = p1[1];
                        ATOMICWRITE xend[this_index] = p2[1];
                        ATOMICWRITE xstart[this_index+1] = p1[2];
                        ATOMICWRITE xend[this_index+1] = p2[2];
                        ATOMICWRITE xstart[this_index+2] = p1[3];
                        ATOMICWRITE xend[this_index+2] = p2[3];
                      }
                  }
                }
              index += p_info.get_num_tangential_poss()*3;
            }
        }
    }

  info("done", 2);
}


END_NAMESPACE_STIR
