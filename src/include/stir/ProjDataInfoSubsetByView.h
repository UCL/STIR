/*
    Copyright (C) ...
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup projdata
  \brief Declaration of class stir::ProjDataInfoSubset

  \author Ashley Gillman
*/
#ifndef __stir_ProjDataInfoCylindrical_H__
#define __stir_ProjDataInfoCylindrical_H__


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
  ProjDataInfoSubsetByView();
  //! Constructor given all the necessary information
  /*! The min and max ring difference in each segment are passed
  as VectorWithOffsets. All three vectors have to have index ranges
  from min_segment_num to max_segment_num.
  
  \warning Most of this library assumes that segment 0 corresponds
  to an average ring difference of 0.
  */
  ProjDataInfoSubsetByView(
    const shared_ptr<ProjDataInfo> full_proj_data_info,
    const /*VectorWithOffset<int>& ?*/ std::vector<int> views);

  ProjDataInfo* clone() const;

  //~ProjDataInfo() {}

  void reduce_segment_range(const int min_segment_num, const int max_segment_num);
 
  //! set new number of views, covering the same azimuthal angle range
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

  inline float get_m(const Bin&) const;

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
  
protected:
  bool blindly_equals(const root_type * const) const;

private:

};


END_NAMESPACE_STIR

//#include "stir/ProjDataInfoSubsetByView.inl"

#endif

