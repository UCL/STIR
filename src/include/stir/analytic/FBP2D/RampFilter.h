//
// $Id$
//
/*!
  \file
  \brief Definition of class RampFilter, used for (2D) FBP
  \ingroup FBP2D
  \author Claire Labbe
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
#ifndef __stir_FBP2D_RampFilter_H__
#define __stir_FBP2D_RampFilter_H__

#ifdef NRFFT
#include "local/stir/Filter.h"
#else
#include "stir/ArrayFilterUsingRealDFTWithPadding.h"
#include "stir/TimedObject.h"
#endif
#include <string>

START_NAMESPACE_STIR
/*!
  \ingroup FBP2D
  \brief The ramp filter used for (2D) FBP
*/
class RampFilter : 
#ifdef NRFFT
  public Filter1D<float>
#else
  public ArrayFilterUsingRealDFTWithPadding<1,float>,
  public TimedObject
#endif
{

private:
  float fc;
  float alpha;
  float sampledist; 
 public:
 RampFilter(float sampledist_v, int length_v , float alpha_v=1, float fc_v=.5); 

 virtual std::string parameter_info() const;
 
};

END_NAMESPACE_STIR

#endif
