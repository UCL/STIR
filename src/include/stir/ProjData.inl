//
// $Id$: $Date$
//
/*!
  \file
  \ingroup buildblock
  \brief Implementations for inline functions of class ProjData

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/

#include "ProjDataInfo.h"

START_NAMESPACE_TOMO

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


END_NAMESPACE_TOMO
