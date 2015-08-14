/*!
  \file
  \ingroup projdata
  \brief Implementations for inline functions of class stir::ProjData

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000-2009, Hammersmith Imanet Ltd
    Copyright (C) 2013, 2015 University College London
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/

#include "stir/ProjDataInfo.h"

START_NAMESPACE_STIR

const ProjDataInfo*
ProjData::get_proj_data_info_ptr() const
{
  return proj_data_info_ptr.get();
}

shared_ptr<ProjDataInfo>
ProjData::get_proj_data_info_sptr() const
{
  return proj_data_info_ptr;
}

const ExamInfo*
ProjData::get_exam_info_ptr() const
{
  return exam_info_sptr.get();
}

shared_ptr<ExamInfo>
ProjData::get_exam_info_sptr() const
{
  return exam_info_sptr;
}

int ProjData::get_num_segments() const
{ return proj_data_info_ptr->get_num_segments(); }

int ProjData::get_num_axial_poss(const int segment_num) const
{ return proj_data_info_ptr->get_num_axial_poss(segment_num); }

int ProjData::get_num_views() const
{ return proj_data_info_ptr->get_num_views(); }

int ProjData::get_num_tangential_poss() const
{ return proj_data_info_ptr->get_num_tangential_poss(); }

int ProjData::get_min_segment_num() const
{ return proj_data_info_ptr->get_min_segment_num(); }

int ProjData::get_max_segment_num() const
{ return proj_data_info_ptr->get_max_segment_num(); }

int ProjData::get_min_axial_pos_num(const int segment_num) const
{ return proj_data_info_ptr->get_min_axial_pos_num(segment_num); }

int ProjData::get_max_axial_pos_num(const int segment_num) const
{ return proj_data_info_ptr->get_max_axial_pos_num(segment_num); }

int ProjData::get_min_view_num() const
{ return proj_data_info_ptr->get_min_view_num(); }

int ProjData::get_max_view_num() const
{ return proj_data_info_ptr->get_max_view_num(); }

int ProjData::get_min_tangential_pos_num() const
{ return proj_data_info_ptr->get_min_tangential_pos_num(); }

int ProjData::get_max_tangential_pos_num() const
{ return proj_data_info_ptr->get_max_tangential_pos_num(); }


END_NAMESPACE_STIR
