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
