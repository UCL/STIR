//
//
/*
  Copyright (C) 2005- 2007, Hammersmith Imanet Ltd
  Copyright 2023, Positrigo AG, Zurich
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup projdata
  \brief Implementation of stir::inverse_SSRB

  \author Charalampos Tsoumpas
  \author Kris Thielemans
  \author Markus Jehl
*/
#include "stir/ProjData.h"
#include "stir/ProjDataInfo.h"
#include "stir/inverse_SSRB.h"
#include "stir/Sinogram.h"
#include "stir/Bin.h"
#include "stir/Succeeded.h"
#include <limits>

START_NAMESPACE_STIR

Succeeded
inverse_SSRB(ProjData& proj_data_4D, const ProjData& proj_data_3D)
{
  const shared_ptr<const ProjDataInfo> proj_data_3D_info_sptr
      = dynamic_pointer_cast<const ProjDataInfo>(proj_data_3D.get_proj_data_info_sptr());
  const shared_ptr<const ProjDataInfo> proj_data_4D_info_sptr
      = dynamic_pointer_cast<const ProjDataInfo>(proj_data_4D.get_proj_data_info_sptr());
  if ((proj_data_3D_info_sptr->get_min_view_num() != proj_data_4D_info_sptr->get_min_view_num())
      || (proj_data_3D_info_sptr->get_min_view_num() != proj_data_4D_info_sptr->get_min_view_num()))
    {
      warning("inverse_SSRB: incompatible view-information");
      return Succeeded::no;
    }
  if ((proj_data_3D_info_sptr->get_min_tangential_pos_num() != proj_data_4D_info_sptr->get_min_tangential_pos_num())
      || (proj_data_3D_info_sptr->get_min_tangential_pos_num() != proj_data_4D_info_sptr->get_min_tangential_pos_num()))
    {
      warning("inverse_SSRB: incompatible tangential_pos-information");
      return Succeeded::no;
    }

  // keep sinograms out of the loop to avoid reallocations
  // initialise to something because there's no default constructor
  Sinogram<float> sino_4D = proj_data_4D.get_empty_sinogram(proj_data_4D.get_min_axial_pos_num(0), 0);
  Sinogram<float> sino_3D_1 = proj_data_3D.get_empty_sinogram(proj_data_3D.get_min_axial_pos_num(0), 0);
  Sinogram<float> sino_3D_2 = proj_data_3D.get_empty_sinogram(proj_data_3D.get_min_axial_pos_num(0), 0);

  // prefill a vector with the axial positions of the direct sinograms
  VectorWithOffset<float> in_m(proj_data_3D.get_min_axial_pos_num(0), proj_data_3D.get_max_axial_pos_num(0));
  for (int in_ax_pos_num = proj_data_3D.get_min_axial_pos_num(0); in_ax_pos_num <= proj_data_3D.get_max_axial_pos_num(0);
       ++in_ax_pos_num)
    {
      in_m.at(in_ax_pos_num) = proj_data_3D_info_sptr->get_m(Bin(0, 0, in_ax_pos_num, 0));
    }

  for (int out_segment_num = proj_data_4D.get_min_segment_num(); out_segment_num <= proj_data_4D.get_max_segment_num();
       ++out_segment_num)
    {
      for (int out_ax_pos_num = proj_data_4D.get_min_axial_pos_num(out_segment_num);
           out_ax_pos_num <= proj_data_4D.get_max_axial_pos_num(out_segment_num); ++out_ax_pos_num)
        {
          for (int k=proj_data_4D.get_proj_data_info_sptr()->get_min_tof_pos_num();
            k<=proj_data_4D.get_proj_data_info_sptr()->get_max_tof_pos_num(); ++k)
          {
            sino_4D = proj_data_4D.get_empty_sinogram(out_ax_pos_num, out_segment_num, false, k);
            const float out_m = proj_data_4D_info_sptr->get_m(Bin(out_segment_num, 0, out_ax_pos_num, 0));

            // Go through all direct sinograms to check which pair are closest.
            bool sinogram_set = false;
            for (int in_ax_pos_num = proj_data_3D.get_min_axial_pos_num(0); in_ax_pos_num <= proj_data_3D.get_max_axial_pos_num(0);
                ++in_ax_pos_num)
              {
                  // for the first slice there is no previous
                  const auto distance_to_previous = in_ax_pos_num == proj_data_3D.get_min_axial_pos_num(0)
                                                        ? std::numeric_limits<float>::max()
                                                        : abs(out_m - in_m.at(in_ax_pos_num - 1));
                  const auto distance_to_current = abs(out_m - in_m.at(in_ax_pos_num));
                  // for the last slice there is no next
                  const auto distance_to_next = in_ax_pos_num == proj_data_3D.get_max_axial_pos_num(0)
                                                    ? std::numeric_limits<float>::max()
                                                    : abs(out_m - in_m.at(in_ax_pos_num + 1));
                  if (distance_to_current <= distance_to_previous && distance_to_current <= distance_to_next)
                    {
                      if (distance_to_current <= 1E-4)
                        {
                          sino_3D_1 = proj_data_3D.get_sinogram(in_ax_pos_num, 0,false, k);
                          sino_4D += sino_3D_1;
                        }
                      else if (distance_to_previous < distance_to_next)
                        { // interpolate between the previous axial slice and this one
                          const auto distance_sum = distance_to_previous + distance_to_current;
                          sino_3D_1 = proj_data_3D.get_sinogram(in_ax_pos_num - 1, 0, false, k);
                          sino_3D_2 = proj_data_3D.get_sinogram(in_ax_pos_num, 0, false, k);
                          sino_3D_1.sapyb(distance_to_current / distance_sum, sino_3D_2, distance_to_previous / distance_sum);
                          sino_4D += sino_3D_1;
                        }
                      else
                        { // interpolate between the next axial slice and this one
                          const auto distance_sum = distance_to_next + distance_to_current;
                          sino_3D_1 = proj_data_3D.get_sinogram(in_ax_pos_num + 1, 0, false, k);
                          sino_3D_2 = proj_data_3D.get_sinogram(in_ax_pos_num, 0, false, k);
                          sino_3D_1.sapyb(distance_to_current / distance_sum, sino_3D_2, distance_to_next / distance_sum);
                          sino_4D += sino_3D_1;
                        }

                      if (proj_data_4D.set_sinogram(sino_4D) == Succeeded::no)
                        return Succeeded::no;
                      sinogram_set = true;
                      break;
                    }
                }
              if (!sinogram_set)
                { // it is logically not possible to get here
                  error("no matching sinogram found for segment %d and axial pos %d", out_segment_num, out_ax_pos_num);
                }
          }  
        }
    }
  return Succeeded::yes;
}
END_NAMESPACE_STIR
