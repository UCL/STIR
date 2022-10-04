//
//
/*!
  \file
  \ingroup Parallelproj

  \brief non-inline implementations for stir::ParallelprojHelper

  \author Kris Thielemans
  
*/
/*
    Copyright (C) 2021 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/recon_buildblock/Parallelproj_projector/ParallelprojHelper.h"
#include "stir/ProjData.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/LORCoordinates.h"
#include "stir/Bin.h"

// for debugging, remove later
#include "stir/info.h"
#include "stir/stream.h"
#include <iostream>


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

  copy_to_array(stir_voxel_size*rescale, voxsize);
  copy_to_array(stir_image.get_lengths(), imgdim);

  BasicCoordinate<3,int> min_indices, max_indices;
  stir_image.get_regular_range(min_indices, max_indices);
  auto coord_first_voxel = stir_image.get_physical_coordinates_for_indices(min_indices);
  // TODO origin shift get_LOR() uses the "centred w.r.t. gantry" coordinate system
  coord_first_voxel[1] -= (stir_image.get_min_index() + stir_image.get_max_index())/2.F * stir_voxel_size[1];
  copy_to_array(coord_first_voxel*rescale, origin);

  // loop over all LORs in the projdata
  Bin bin;
  LORInAxialAndNoArcCorrSinogramCoordinates<float> lor;
  LORAs2Points<float> lor_points;
  const float radius = p_info.get_scanner_sptr()->get_max_FOV_radius() + p_info.get_scanner_sptr()->get_average_depth_of_interaction();

  // warning: next loop needs to be the same as how ProjDataInMemory stores its data. There is no guarantee that this will remain the case in the future.
  const auto segment_sequence = ProjData::standard_segment_sequence(p_info);
  std::size_t index(0);
  for (int seg : segment_sequence)
    {
      bin.segment_num() = seg;
      for (bin.axial_pos_num() = p_info.get_min_axial_pos_num(bin.segment_num()); bin.axial_pos_num() <= p_info.get_max_axial_pos_num(bin.segment_num()); ++bin.axial_pos_num())
        {
          for (bin.view_num() = p_info.get_min_view_num(); bin.view_num() <= p_info.get_max_view_num(); ++bin.view_num())
            {
              for (bin.tangential_pos_num() = p_info.get_min_tangential_pos_num(); bin.tangential_pos_num() <= p_info.get_max_tangential_pos_num(); ++bin.tangential_pos_num())
                {
                  p_info.get_LOR(lor, bin);
                  lor.get_intersections_with_cylinder(lor_points, radius);
                  const CartesianCoordinate3D<float> p1 = lor_points.p1()*rescale;
                  const CartesianCoordinate3D<float> p2 = lor_points.p2()*rescale;
#if 0
                  if (index+2 > xstart.size())
                    error("That went wrong: index " + std::to_string(index) + ", xstart size " + std::to_string(xstart.size()));
                  if (index+2 > xend.size())
                    error("That went wrong: index " + std::to_string(index) + ", xend size " + std::to_string(xstart.size()));
#endif
                  xstart[index] = p1[1];
                  xend[index++] = p2[1];
                  xstart[index] = p1[2];
                  xend[index++] = p2[2];
                  xstart[index] = p1[3];
                  xend[index++] = p2[3];
                }
            }
        }
    }

  info("done", 2);
}


END_NAMESPACE_STIR
