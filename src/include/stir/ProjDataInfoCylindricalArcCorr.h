//
// $Id$
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2007-10-08, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - $Date$, Kris Thielemans
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
/*!

  \file
  \ingroup projdata

  \brief Declaration of class stir::ProjDataInfoCylindricalArcCorr

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project

  $Date$

  $Revision$
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
  ProjDataInfoCylindricalArcCorr(const shared_ptr<Scanner> scanner_ptr,float bin_size,
    const  VectorWithOffset<int>& num_axial_pos_per_segment,
    const  VectorWithOffset<int>& min_ring_diff_v, 
    const  VectorWithOffset<int>& max_ring_diff_v,
    const int num_views,const int num_tangential_poss);

  ProjDataInfo* clone() const;
  
  bool operator==(const self_type&) const;

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

  virtual bool blindly_equals(const root_type * const) const;

};

END_NAMESPACE_STIR

#include "stir/ProjDataInfoCylindricalArcCorr.inl"

#endif
