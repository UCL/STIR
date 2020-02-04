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

//const ExamInfo*
//ProjData::get_exam_info_ptr() const
//{
//  return exam_info_sptr.get();
//}

//shared_ptr<ExamInfo>
//ProjData::get_exam_info_sptr() const
//{
//  return exam_info_sptr;
//}

int ProjData::get_num_segments() const
{ return proj_data_info_ptr->get_num_segments(); }

int ProjData::get_num_axial_poss(const int segment_num) const
{ return proj_data_info_ptr->get_num_axial_poss(segment_num); }

int ProjData::get_num_views() const
{ return proj_data_info_ptr->get_num_views(); }

int ProjData::get_num_tangential_poss() const
{ return proj_data_info_ptr->get_num_tangential_poss(); }

int ProjData::get_num_tof_poss() const
{ return proj_data_info_ptr->get_num_tof_poss(); }

int ProjData::get_tof_mash_factor() const
{ return proj_data_info_ptr->get_tof_mash_factor(); }

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

int ProjData::get_min_tof_pos_num() const
{ return proj_data_info_ptr->get_min_tof_pos_num(); }

int ProjData::get_max_tof_pos_num() const
{ return proj_data_info_ptr->get_max_tof_pos_num(); }

int ProjData::get_num_non_tof_sinograms() const
{
    int num_sinos = proj_data_info_ptr->get_num_axial_poss(0);
    for (int s=1; s<= this->get_max_segment_num(); ++s)
        num_sinos += 2* this->get_num_axial_poss(s);

    return num_sinos;
}

int ProjData::get_num_sinograms() const
{
    return this->get_num_non_tof_sinograms()*this->get_num_tof_poss();
}

std::size_t ProjData::size_all() const
{ return this->get_num_sinograms() * this->get_num_views() * this->get_num_tangential_poss(); }

END_NAMESPACE_STIR
