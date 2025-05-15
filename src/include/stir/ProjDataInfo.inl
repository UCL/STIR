//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2011-10-14, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2011, Kris Thielemans
    Copyright (C) 2016, University of Hull
    Copyright (C) 2016, 2017, 2020, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license
    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup projdata
  \brief Implementations of inline functions for class stir::ProjDataInfo

  \author Nikos Efthimiou
  \author Sanida Mustafovic
  \author Kris Thielemans
  \author Elise Emond
  \author Nikos Efthimiou
  \author PARAPET project

*/

#include "stir/format.h"
#include "stir/warning.h"
START_NAMESPACE_STIR

shared_ptr<ProjDataInfo>
ProjDataInfo::create_shared_clone() const
{
  shared_ptr<ProjDataInfo> sptr(this->clone());
  return sptr;
}

shared_ptr<ProjDataInfo>
ProjDataInfo::create_non_tof_clone() const
{
  shared_ptr<ProjDataInfo> sptr(this->clone());
  sptr->set_tof_mash_factor(0); // tof mashing factor = 0 is a trigger for non-tof data
  return sptr;
}

int
ProjDataInfo::get_num_segments() const
{
  return (max_axial_pos_per_seg.get_length());
}

int
ProjDataInfo::get_num_axial_poss(const int segment_num) const
{
  return max_axial_pos_per_seg[segment_num] - min_axial_pos_per_seg[segment_num] + 1;
}

int
ProjDataInfo::get_num_views() const
{
  return max_view_num - min_view_num + 1;
}

int
ProjDataInfo::get_num_tangential_poss() const
{
  return max_tangential_pos_num - min_tangential_pos_num + 1;
}

int
ProjDataInfo::get_num_tof_poss() const
{
  return num_tof_bins;
}

int
ProjDataInfo::get_tof_bin(const double delta) const
{
  if (!is_tof_data())
    return 0;

  for (int i = min_tof_pos_num; i <= max_tof_pos_num; ++i)
    {
      if (delta >= tof_bin_boundaries_ps[i].low_lim && delta < tof_bin_boundaries_ps[i].high_lim)
        return i;
    }
  // TODO handle differently
  warning(format("TOF delta time {} out of range", delta));
  return min_tof_pos_num;
}

#if 0
// KT: code disabled as buggy but currently not needed
int
ProjDataInfo::get_unmashed_tof_bin(const double delta) const
{
  if (!is_tof_data())
    return 0;

  for (int i = min_unmashed_tof_pos_num; i <= max_unmashed_tof_pos_num; ++i)
  {
    if (delta >= tof_bin_boundaries_ps[i].low_lim &&
      delta < tof_bin_boundaries_ps[i].high_lim)
      return i;
  }
  // TODO handle differently
  warning(format("TOF delta time {} out of range", delta));
  return min_tof_pos_num;

}
#endif

int
ProjDataInfo::get_tof_mash_factor() const
{
  return tof_mash_factor;
}

int
ProjDataInfo::get_min_segment_num() const
{
  return (max_axial_pos_per_seg.get_min_index());
}

int
ProjDataInfo::get_max_segment_num() const
{
  return (max_axial_pos_per_seg.get_max_index());
}

int
ProjDataInfo::get_min_axial_pos_num(const int segment_num) const
{
  return min_axial_pos_per_seg[segment_num];
}

int
ProjDataInfo::get_max_axial_pos_num(const int segment_num) const
{
  return max_axial_pos_per_seg[segment_num];
}

int
ProjDataInfo::get_min_view_num() const
{
  return min_view_num;
}

int
ProjDataInfo::get_max_view_num() const
{
  return max_view_num;
}

int
ProjDataInfo::get_min_tangential_pos_num() const
{
  return min_tangential_pos_num;
}

int
ProjDataInfo::get_max_tangential_pos_num() const
{
  return max_tangential_pos_num;
}

int
ProjDataInfo::get_min_tof_pos_num() const
{
  return min_tof_pos_num;
}

int
ProjDataInfo::get_max_tof_pos_num() const
{
  return max_tof_pos_num;
}

bool
ProjDataInfo::is_tof_data() const
{
  // First case: if tof_mash_factor == 0, scanner is not tof ready and no tof data
  if (tof_mash_factor == 0)
    {
      if (num_tof_bins != 1)
        {
          error("Non-TOF data with inconsistent Time-of-Flight bin number - Aborted operation.");
        }
      return false;
    }
  // Second case: when tof_mash_factor is strictly positive, it means we have TOF data
  else if (tof_mash_factor > 0)
    {
      return true;
    }
  return false;
}

float
ProjDataInfo::get_costheta(const Bin& bin) const
{
  return 1 / sqrt(1 + square(get_tantheta(bin)));
}

float
ProjDataInfo::get_m(const Bin& bin) const
{
  return get_t(bin) / get_costheta(bin);
}

const Scanner*
ProjDataInfo::get_scanner_ptr() const
{
  return scanner_ptr.get();
}

shared_ptr<Scanner>
ProjDataInfo::get_scanner_sptr() const
{
  return scanner_ptr;
}

int
ProjDataInfo::get_num_non_tof_sinograms() const
{
  int num_sinos = 0;
  for (int s = this->get_min_segment_num(); s <= this->get_max_segment_num(); ++s)
    num_sinos += this->get_num_axial_poss(s);

  return num_sinos;
}

int
ProjDataInfo::get_num_sinograms() const
{
  return this->get_num_non_tof_sinograms() * this->get_num_tof_poss();
}

std::size_t
ProjDataInfo::size_all() const
{
  return static_cast<std::size_t>(this->get_num_sinograms())
         * static_cast<std::size_t>(this->get_num_views() * this->get_num_tangential_poss());
}

END_NAMESPACE_STIR
