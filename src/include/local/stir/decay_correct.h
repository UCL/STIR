//
// $Id$
//
/*
    Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
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
#ifndef __stir_decay_correct_H__
#define __stir_decay_correct_H__
/*!
  \file 
  \brief Implementation of simple function to provide the decay factor. 

  \author Charalampos Tsoumpas
  \author Kris Thielemans

  $Date$
  $Revision$
*/

#include "stir/common.h"

START_NAMESPACE_STIR

//! This function uses the approximation: C(t)*(t2-t1)=C0*Integral[t1->t2](2^(-t/halftime)dt), in order to find C0.
//  For FDG is not necessary to use this approximation, because it will be slower. 
inline double
decay_correct_factor(const double isotope_halflife, const double start_time, const double end_time)  
{ 
  assert(end_time-start_time>0.001);
  return std::log(2.)*(end_time-start_time)/(isotope_halflife
	 *(std::exp(-start_time*std::log(2.)/isotope_halflife)-std::exp(-end_time*std::log(2.)/isotope_halflife)));
}
//! This function uses the approximation: C(t)=C0*2^(-t/halftime), in order to find C0.
//For FDG is fine to use this approximation, becaus it will be slower. 
inline double decay_correct_factor(const double isotope_halflife, const double mean_time)
{ 
  assert(mean_time>0.001); 
  return   std::exp(mean_time*std::log(2.)/isotope_halflife); 
}

END_NAMESPACE_STIR

#endif
