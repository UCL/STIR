/*
    Copyright (C) 2000, Hammersmith Imanet Ltd
    Copyright (C) 2016, 2026 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup buildblock
  \brief inline implementations of stir::DataWithProjDataInfo

  \author Kris Thielemans
*/

START_NAMESPACE_STIR

int
DataWithProjDataInfo::get_num_segments() const
{
  return proj_data_info_sptr->get_num_segments();
}

int
DataWithProjDataInfo::get_num_axial_poss(const int segment_num) const
{
  return proj_data_info_sptr->get_num_axial_poss(segment_num);
}

int
DataWithProjDataInfo::get_num_views() const
{
  return proj_data_info_sptr->get_num_views();
}

int
DataWithProjDataInfo::get_num_tangential_poss() const
{
  return proj_data_info_sptr->get_num_tangential_poss();
}

int
DataWithProjDataInfo::get_num_tof_poss() const
{
  return proj_data_info_sptr->get_num_tof_poss();
}

int
DataWithProjDataInfo::get_tof_mash_factor() const
{
  return proj_data_info_sptr->get_tof_mash_factor();
}

int
DataWithProjDataInfo::get_min_segment_num() const
{
  return proj_data_info_sptr->get_min_segment_num();
}

int
DataWithProjDataInfo::get_max_segment_num() const
{
  return proj_data_info_sptr->get_max_segment_num();
}

int
DataWithProjDataInfo::get_min_axial_pos_num(const int segment_num) const
{
  return proj_data_info_sptr->get_min_axial_pos_num(segment_num);
}

int
DataWithProjDataInfo::get_max_axial_pos_num(const int segment_num) const
{
  return proj_data_info_sptr->get_max_axial_pos_num(segment_num);
}

int
DataWithProjDataInfo::get_min_view_num() const
{
  return proj_data_info_sptr->get_min_view_num();
}

int
DataWithProjDataInfo::get_max_view_num() const
{
  return proj_data_info_sptr->get_max_view_num();
}

int
DataWithProjDataInfo::get_min_tangential_pos_num() const
{
  return proj_data_info_sptr->get_min_tangential_pos_num();
}

int
DataWithProjDataInfo::get_max_tangential_pos_num() const
{
  return proj_data_info_sptr->get_max_tangential_pos_num();
}

int
DataWithProjDataInfo::get_min_tof_pos_num() const
{
  return proj_data_info_sptr->get_min_tof_pos_num();
}

int
DataWithProjDataInfo::get_max_tof_pos_num() const
{
  return proj_data_info_sptr->get_max_tof_pos_num();
}

END_NAMESPACE_STIR
