//
// $Id$
//
/*!

  \file
  \ingroup projdata

  \brief Declaration of class ProjDataInfoCylindricalArcCorr

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project

  $Date$

  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#ifndef __ProjDataInfoCylindricalArcCorr_H__
#define __ProjDataInfoCylindricalArcCorr_H__


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

public:
  //! Constructors
  ProjDataInfoCylindricalArcCorr();
  ProjDataInfoCylindricalArcCorr(const shared_ptr<Scanner> scanner_ptr,float bin_size,
    const  VectorWithOffset<int>& num_axial_pos_per_segment,
    const  VectorWithOffset<int>& min_ring_diff_v, 
    const  VectorWithOffset<int>& max_ring_diff_v,
    const int num_views,const int num_tangential_poss);

  ProjDataInfo* clone() const;
  
  inline virtual float get_s(const Bin&) const;
  //! Set tangential sampling
  void set_tangential_sampling(const float bin_size);
  //! Get tangential sampling
  inline float get_tangential_sampling() const;
  virtual float get_sampling_in_s(const Bin&) const
  {return bin_size; }

  virtual 
    Bin
    get_bin(const LOR<float>&) const;

  virtual string parameter_info() const;
private:
  
  float bin_size;
  

};

END_NAMESPACE_STIR

#include "stir/ProjDataInfoCylindricalArcCorr.inl"

#endif
