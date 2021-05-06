//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup numerics
  \brief implementation of stir::divide

  \author Matthew Jacobson
  \author Kris Thielemans
  \author PARAPET project
  
*/
#include <cmath>

START_NAMESPACE_STIR

template <class NumeratorIterT,
	  class DenominatorIterT,
	  class small_numT>
void 
divide(
       const NumeratorIterT& numerator_begin,
       const NumeratorIterT& numerator_end,
       const DenominatorIterT& denominator_begin,
       const small_numT small_num)
{
  small_numT small_value= 
    *std::max_element(numerator_begin, numerator_end)
    *small_num;
  small_value=(small_value>0)?small_value:0;   

  NumeratorIterT numerator_iter = numerator_begin;
  DenominatorIterT denominator_iter = denominator_begin;
  while (numerator_iter != numerator_end)
    {
      if(std::fabs(*denominator_iter)<=small_value && std::fabs(*numerator_iter)<=small_value) 
	  (*numerator_iter)=0;
      else 
	(*numerator_iter)/=(*denominator_iter);
      ++numerator_iter;
      ++denominator_iter;
    }
}

END_NAMESPACE_STIR
