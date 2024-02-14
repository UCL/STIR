/*!
  \file
  \ingroup projdata
  \brief Implementations for inline functions of class stir::ProjData

  \author Nikos Efthimiou
  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000-2009, Hammersmith Imanet Ltd
    Copyright (C) 2013, 2015, 2023 University College London
    Copyright (C) 2016, University of Hull
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/

#include "stir/ProjDataInfo.h"

START_NAMESPACE_STIR

SegmentBySinogram<float>
ProjData::get_segment_by_sinogram(const SegmentIndices& si) const
{
  return this->get_segment_by_sinogram(si.segment_num(), si.timing_pos_num());
}

SegmentByView<float>
ProjData::get_segment_by_view(const SegmentIndices& si) const
{
  return this->get_segment_by_view(si.segment_num(), si.timing_pos_num());
}

Viewgram<float>
ProjData::get_viewgram(const ViewgramIndices& vi) const
{
  return this->get_viewgram(vi.view_num(), vi.segment_num(), false, vi.timing_pos_num());
}

Sinogram<float>
ProjData::get_sinogram(const SinogramIndices& vi) const
{
  return this->get_sinogram(vi.axial_pos_num(), vi.segment_num(), false, vi.timing_pos_num());
}

shared_ptr<const ProjDataInfo>
ProjData::get_proj_data_info_sptr() const
{
  return proj_data_info_sptr;
}

int
ProjData::get_num_segments() const
{
  return proj_data_info_sptr->get_num_segments();
}

int
ProjData::get_num_axial_poss(const int segment_num) const
{
  return proj_data_info_sptr->get_num_axial_poss(segment_num);
}

int
ProjData::get_num_views() const
{
  return proj_data_info_sptr->get_num_views();
}

int
ProjData::get_num_tangential_poss() const
{
  return proj_data_info_sptr->get_num_tangential_poss();
}

int
ProjData::get_num_tof_poss() const
{
  return proj_data_info_sptr->get_num_tof_poss();
}

int
ProjData::get_tof_mash_factor() const
{
  return proj_data_info_sptr->get_tof_mash_factor();
}

int
ProjData::get_min_segment_num() const
{
  return proj_data_info_sptr->get_min_segment_num();
}

int
ProjData::get_max_segment_num() const
{
  return proj_data_info_sptr->get_max_segment_num();
}

int
ProjData::get_min_axial_pos_num(const int segment_num) const
{
  return proj_data_info_sptr->get_min_axial_pos_num(segment_num);
}

int
ProjData::get_max_axial_pos_num(const int segment_num) const
{
  return proj_data_info_sptr->get_max_axial_pos_num(segment_num);
}

int
ProjData::get_min_view_num() const
{
  return proj_data_info_sptr->get_min_view_num();
}

int
ProjData::get_max_view_num() const
{
  return proj_data_info_sptr->get_max_view_num();
}

int
ProjData::get_min_tangential_pos_num() const
{
  return proj_data_info_sptr->get_min_tangential_pos_num();
}

int
ProjData::get_max_tangential_pos_num() const
{
  return proj_data_info_sptr->get_max_tangential_pos_num();
}

int
ProjData::get_min_tof_pos_num() const
{
  return proj_data_info_sptr->get_min_tof_pos_num();
}

int
ProjData::get_max_tof_pos_num() const
{
  return proj_data_info_sptr->get_max_tof_pos_num();
}

int
ProjData::get_num_non_tof_sinograms() const
{
  return proj_data_info_sptr->get_num_non_tof_sinograms();
}

int
ProjData::get_num_sinograms() const
{
  return proj_data_info_sptr->get_num_sinograms();
}

std::size_t
ProjData::size_all() const
{
  return proj_data_info_sptr->size_all();
}

std::vector<int>
ProjData::get_original_view_nums() const
{
  return proj_data_info_sptr->get_original_view_nums();
}

END_NAMESPACE_STIR
