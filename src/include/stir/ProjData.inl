//
// $Id$
//
/*!
  \file
  \ingroup projdata
  \brief Implementations for inline functions of class stir::ProjData

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project

  $Date$

  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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

ProjData:: ProjData()
{}

ProjData::ProjData(const shared_ptr<ProjDataInfo>& proj_data_info_ptr)
: proj_data_info_ptr(proj_data_info_ptr)
{}


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
