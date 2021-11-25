//
//
/*
    Copyright (C) 2021, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup projdata

  \brief Implementation of non-inline functions of class stir::ProjDataInfoSubsetByView

  \author Kris Thielemans

*/
#include "stir/ProjDataInfoSubsetByView.h"
#include "stir/Bin.h"

START_NAMESPACE_STIR

ProjDataInfoSubsetByView::ProjDataInfoSubsetByView(const shared_ptr<const ProjDataInfo> full_proj_data_info_sptr,
                                                   const std::vector<int>& views)
  :
  org_proj_data_info_sptr(full_proj_data_info_sptr->clone()),
  view_to_org_view_num(views)
{
  // TODO initialise the org_view_to_view_num
}


ProjDataInfoSubsetByView* ProjDataInfoSubsetByView::clone() const
{
  return new ProjDataInfoSubsetByView(this->org_proj_data_info_sptr, this->org_view_to_view_num);
}

Bin ProjDataInfoSubsetByView::get_org_bin(const Bin& bin) const
{
  Bin org_bin(bin);
  org_bin.view_num() = this->view_to_org_view_num[bin.view_num()];
  return org_bin;
}


Bin ProjDataInfoSubsetByView::get_bin_from_org(const Bin& org_bin) const
{
  Bin bin(org_bin);
  bin.view_num() = this->org_view_to_view_num[org_bin.view_num()];
  return bin;
}

void ProjDataInfoSubsetByView::reduce_segment_range(const int min_segment_num, const int max_segment_num)
{
  this->org_proj_data_info_sptr->reduce_segment_range(min_segment_num, max_segment_num);
}
 
void ProjDataInfoSubsetByView::set_num_views(const int)
{
  error("ProjDataInfoSubsetByView does not allow changing the number of views");
}

void ProjDataInfoSubsetByView::set_num_tangential_poss(const int num_tang_poss)
{
  this->org_proj_data_info_sptr->set_num_tangential_poss(num_tang_poss);
}

void ProjDataInfoSubsetByView::set_num_axial_poss_per_segment(const VectorWithOffset<int>& num_axial_poss_per_segment)
{
  this->org_proj_data_info_sptr->set_num_axial_poss_per_segment(num_axial_poss_per_segment);
}

void ProjDataInfoSubsetByView::set_min_axial_pos_num(const int min_ax_pos_num, const int segment_num)
{
  this->org_proj_data_info_sptr->set_min_axial_pos_num(min_ax_pos_num, segment_num);
}

void ProjDataInfoSubsetByView::set_max_axial_pos_num(const int max_ax_pos_num, const int segment_num)
{
  this->org_proj_data_info_sptr->set_max_axial_pos_num(max_ax_pos_num, segment_num);
}

void ProjDataInfoSubsetByView::set_min_tangential_pos_num(const int min_tang_poss)
{
  this->org_proj_data_info_sptr->set_min_tangential_pos_num(min_tang_poss);
}

void ProjDataInfoSubsetByView::set_max_tangential_pos_num(const int max_tang_poss)
{
  this->org_proj_data_info_sptr->set_max_tangential_pos_num(max_tang_poss);
}

float ProjDataInfoSubsetByView::get_tantheta(const Bin& bin) const
{
  return this->org_proj_data_info_sptr->get_tantheta(get_org_bin(bin));
}

float ProjDataInfoSubsetByView::get_phi(const Bin& bin) const
{
  return this->org_proj_data_info_sptr->get_phi(get_org_bin(bin));
}

float ProjDataInfoSubsetByView::get_t(const Bin& bin) const
{
  return this->org_proj_data_info_sptr->get_t(get_org_bin(bin));
}

float ProjDataInfoSubsetByView::get_m(const Bin& bin) const
{
  return this->org_proj_data_info_sptr->get_m(get_org_bin(bin));
}

float ProjDataInfoSubsetByView::get_s(const Bin& bin) const
{
  return this->org_proj_data_info_sptr->get_s(get_org_bin(bin));
}

void ProjDataInfoSubsetByView::get_LOR(LORInAxialAndNoArcCorrSinogramCoordinates<float>& lor,
                                       const Bin& bin) const
{
  return this->org_proj_data_info_sptr->get_LOR(lor, get_org_bin(bin));
}
                                       
float ProjDataInfoSubsetByView::get_sampling_in_t(const Bin& bin) const
{
  return this->org_proj_data_info_sptr->get_sampling_in_t(get_org_bin(bin));
}

float ProjDataInfoSubsetByView::get_sampling_in_m(const Bin& bin) const
{
  return this->org_proj_data_info_sptr->get_sampling_in_m(get_org_bin(bin));
}

float ProjDataInfoSubsetByView::get_sampling_in_s(const Bin& bin) const
{
  return this->org_proj_data_info_sptr->get_sampling_in_s(get_org_bin(bin));
}

Bin ProjDataInfoSubsetByView::get_bin(const LOR<float>& lor) const
{
  return get_bin_from_org(this->org_proj_data_info_sptr->get_bin(lor));
}

bool ProjDataInfoSubsetByView::operator>=(const ProjDataInfo& proj) const
{
  // TODO compare view table
  return this->org_proj_data_info_sptr->operator>=(proj);
}

std::string ProjDataInfoSubsetByView::parameter_info() const
{
  // TODO insert view info
  return this->org_proj_data_info_sptr->parameter_info();
}
  
bool ProjDataInfoSubsetByView::blindly_equals(const root_type * const p) const
{
  if ((*this->org_proj_data_info_sptr) != (*p))
    return false;
  // TODO compare view tables
  return true;
}

END_NAMESPACE_STIR
