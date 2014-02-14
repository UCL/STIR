//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_divide_H__
#define __stir_divide_H__
/*!
  \file
  \ingroup numerics
  \brief implementation of stir::divide

  \author Matthew Jacobson
  \author Kris Thielemans
  \author PARAPET project
  
*/

START_NAMESPACE_STIR

//! division of two ranges, 0/0 = 0
/*! 
  \ingroup numerics

  This function sets 0/0 to 0 (not the usual NaN). It is for instance useful in
  Poisson log-likelihood computation.

  Because of potential numerical rounding problems, we test if a number is 
  0 by comparing its absolute value with a small value, which is determined
  by multiplying the maximum in the \c numerator range with \a small_num.

  \warning This function does not test for non-zero numbers by 0. Results in 
  that case will likely depend on your processor and/or compiler settings.
*/
template <class NumeratorIterT,
	  class DenominatorIterT,
	  class small_numT>
inline
void 
divide(const NumeratorIterT& numerator_begin,
       const NumeratorIterT& numerator_end,
       const DenominatorIterT& denominator_begin,
       const small_numT small_num);

END_NAMESPACE_STIR

#include "stir/numerics/divide.inl"

#endif
