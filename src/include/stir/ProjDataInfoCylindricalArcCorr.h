//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2007-10-08, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2011, Kris Thielemans
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup projdata

  \brief Declaration of class stir::ProjDataInfoCylindricalArcCorr

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project


*/
#ifndef __stir_ProjDataInfoCylindricalArcCorr_H__
#define __stir_ProjDataInfoCylindricalArcCorr_H__

#include "stir/ProjDataInfoCylindrical.h"

START_NAMESPACE_STIR

/*!
  \ingroup projdata
  \brief Projection data info for arc-corrected data

  This means that 'tangential_pos_num' actually indexes a linear coordinate
  with a particular sampling distance (usually called the 'bin_size').
  */
class ProjDataInfoCylindricalArcCorr : public ProjDataInfoCylindrical
{
  typedef ProjDataInfoCylindrical base_type;
#ifdef SWIG
  // SWIG needs this typedef to be public
 public:
#endif
  typedef ProjDataInfoCylindricalArcCorr self_type;

public:
  //! Constructors
  ProjDataInfoCylindricalArcCorr();
  ProjDataInfoCylindricalArcCorr(const shared_ptr<Scanner> scanner_ptr,
                                 float bin_size,
                                 const VectorWithOffset<int>& num_axial_pos_per_segment,
                                 const VectorWithOffset<int>& min_ring_diff_v,
                                 const VectorWithOffset<int>& max_ring_diff_v,
                                 const int num_views,
                                 const int num_tangential_poss,
                                 const int tof_mash_factor = 0);

  ProjDataInfo* clone() const override;

  bool operator==(const self_type&) const;

  inline float get_s(const Bin&) const override;
  //! Set tangential sampling
  void set_tangential_sampling(const float bin_size);
  //! Get tangential sampling
  inline float get_tangential_sampling() const;
  float get_sampling_in_s(const Bin&) const override
  {
    return bin_size;
  }

  Bin get_bin(const LOR<float>&, const double delta_time = 0.0) const override;

  std::string parameter_info() const override;

private:
  float bin_size;

  bool blindly_equals(const root_type* const) const override;
};

END_NAMESPACE_STIR

#include "stir/ProjDataInfoCylindricalArcCorr.inl"

#endif
