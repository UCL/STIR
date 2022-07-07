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
    Copyright (C) 2013, 2015 University College London
    Copyright (C) 2016, University of Hull
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/

#include "stir/ProjDataInfo.h"

START_NAMESPACE_STIR

shared_ptr<const ProjDataInfo>
ProjData::get_proj_data_info_sptr() const
{
  return proj_data_info_sptr;
}

int ProjData::get_num_segments() const
{ return proj_data_info_sptr->get_num_segments(); }

int ProjData::get_num_axial_poss(const int segment_num) const
{ return proj_data_info_sptr->get_num_axial_poss(segment_num); }

int ProjData::get_num_views() const
{ return proj_data_info_sptr->get_num_views(); }

int ProjData::get_num_tangential_poss() const
{ return proj_data_info_sptr->get_num_tangential_poss(); }

int ProjData::get_num_tof_poss() const
{ return proj_data_info_sptr->get_num_tof_poss(); }

int ProjData::get_min_segment_num() const
{ return proj_data_info_sptr->get_min_segment_num(); }

int ProjData::get_max_segment_num() const
{ return proj_data_info_sptr->get_max_segment_num(); }

int ProjData::get_min_axial_pos_num(const int segment_num) const
{ return proj_data_info_sptr->get_min_axial_pos_num(segment_num); }

int ProjData::get_max_axial_pos_num(const int segment_num) const
{ return proj_data_info_sptr->get_max_axial_pos_num(segment_num); }

int ProjData::get_min_view_num() const
{ return proj_data_info_sptr->get_min_view_num(); }

int ProjData::get_max_view_num() const
{ return proj_data_info_sptr->get_max_view_num(); }

int ProjData::get_min_tangential_pos_num() const
{ return proj_data_info_sptr->get_min_tangential_pos_num(); }

int ProjData::get_max_tangential_pos_num() const
{ return proj_data_info_sptr->get_max_tangential_pos_num(); }

int ProjData::get_num_non_tof_sinograms() const
{ return proj_data_info_sptr->get_num_non_tof_sinograms(); }

int ProjData::get_num_sinograms() const
{ return proj_data_info_sptr->get_num_sinograms(); }

std::size_t ProjData::size_all() const
{ return proj_data_info_sptr->size_all(); }

std::vector<int> ProjData::get_original_view_nums() const
{ return proj_data_info_sptr->get_original_view_nums(); }
  
END_NAMESPACE_STIR
