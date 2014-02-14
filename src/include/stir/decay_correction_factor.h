//
//
/*
    Copyright (C) 2005- 2009, Hammersmith Imanet Ltd
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
#ifndef __stir_decay_correction_factor_H__
#define __stir_decay_correction_factor_H__
/*!
  \file 
  \ingroup buildblock
  \brief Simple functions to compute the decay correction factor. 

  \author Charalampos Tsoumpas
  \author Kris Thielemans

*/

#include "stir/common.h"
#include <cmath>

START_NAMESPACE_STIR

//! Compute decay-correction factor for a time frame
/*!
   \ingroup buildblock 
   This function computes the factor eneded to convert <i>average number of counts per second</i> to
   <i>activity at time 0</i>, i.e. it returns
   \f[ \frac{(t_2-t_1)}{ \int_{t_1}^{t_2} \! 2^{-t/\mathrm{halflife}} \, dt} \f]
 */
inline double
decay_correction_factor(const double isotope_halflife, const double start_time, const double end_time)  
{ 
  assert(end_time-start_time>0);
  const double lambda=std::log(2.)/isotope_halflife;

  return 
    std::fabs(lambda*(end_time-start_time)) < .01
    ? std::exp(-start_time*lambda) // if very short frame, we can ignore the duration
    : lambda*(end_time-start_time)/
      (std::exp(-start_time*lambda)-std::exp(-end_time*lambda));
}

//! Computes the decay-correction factor for activity at a given time point
/*! \ingroup buildblock
  This function computes the correction factor to convert activity at t0 + \a rel_time to activity at t0, i.e.
  \f[ 2^{(\mathrm{rel\_time} / \mathrm{halflife})} \f]
*/
inline double decay_correction_factor(const double isotope_halflife, const double rel_time)
{ 
  return std::exp(rel_time*std::log(2.)/isotope_halflife); 
}

END_NAMESPACE_STIR

#endif
