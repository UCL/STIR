/*
    Copyright (C) ...
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup projdata
  \brief Declaration of class stir::ProjDataInfoSubsetByView

  \author Ashley Gillman
*/
#ifndef __stir_ProjDataInfoSubsetByView__H__
#define __stir_ProjDataInfoSubsetByView__H__


#include "stir/ProjDataInfo.h"
#include <utility>
#include <vector>

START_NAMESPACE_STIR

class Succeeded;

/*!
  \ingroup projdata 
  \brief projection data info for data corresponding to a 
  'subset' sampling, subset by views.
*/
// TODOdoc more
class ProjDataInfoSubsetByView: public ProjDataInfo
{
private:
  typedef ProjDataInfo base_type;
  typedef ProjDataInfoSubsetByView self_type;

public:
  //! Constructors
  //ProjDataInfoSubsetByView();
  ProjDataInfoSubsetByView(const shared_ptr<const ProjDataInfo> full_proj_data_info_sptr,
                           const std::vector<int>& views);

  ProjDataInfoSubsetByView* clone() const;

  //! true if the subset is actually all of the data
  bool contains_full_data() const;

  Bin get_org_bin(const Bin& bin) const;
  Bin get_bin_from_org(const Bin& org_bin) const;

  void reduce_segment_range(const int min_segment_num, const int max_segment_num);
 
  //! this will call error()
  void
    set_num_views(const int new_num_views);

  void set_num_tangential_poss(const int num_tang_poss);

  void set_num_axial_poss_per_segment(const VectorWithOffset<int>& num_axial_poss_per_segment); 

  void set_min_axial_pos_num(const int min_ax_pos_num, const int segment_num);

  void set_max_axial_pos_num(const int max_ax_pos_num, const int segment_num);

  void set_min_tangential_pos_num(const int min_tang_poss);

  void set_max_tangential_pos_num(const int max_tang_poss);

  float get_tantheta(const Bin&) const;

  float get_phi(const Bin&) const;

  float get_t(const Bin&) const;

  float get_m(const Bin&) const;

  float get_s(const Bin&) const;

  void
    get_LOR(LORInAxialAndNoArcCorrSinogramCoordinates<float>&,
	    const Bin&) const;

  float get_sampling_in_t(const Bin&) const;

  float get_sampling_in_m(const Bin&) const;

  float get_sampling_in_s(const Bin&) const;

  Bin get_bin(const LOR<float>&) const;

  bool operator>=(const ProjDataInfo& proj) const;

  std::string parameter_info() const;

  shared_ptr<const ProjDataInfo> get_org_proj_data_info_sptr() const;

protected:
  bool blindly_equals(const root_type * const) const;

private:

  shared_ptr<ProjDataInfo> org_proj_data_info_sptr;
  std::vector<int> view_to_org_view_num;
  std::vector<int> org_view_to_view_num;
};


END_NAMESPACE_STIR

//#include "stir/ProjDataInfoSubsetByView.inl"

#endif

