//
// $Id$
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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
/*!
  \file
  \ingroup numerics
  \brief implementation of stir::divide

  \author Matthew Jacobson
  \author Kris Thielemans
  \author PARAPET project
  
  $Date$
  $Revision$
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
