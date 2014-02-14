//
//
/*
    Copyright (C) 2002- 2007, Hammersmith Imanet Ltd
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

#ifndef __stir_min_positive_element_h_
#define __stir_min_positive_element_h_

/*!
  \file 
  \ingroup buildblock
 
  \brief Declares the stir::min_positive_element() function

  \author Kris Thielemans

*/

#include "stir/common.h"
#include <iostream>
START_NAMESPACE_STIR

//! Finds where the smallest strictly positive element occurs
/*!
   \ingroup buildblock
   \param start start of the sequence. Usually object.begin().
   \param end end of the sequence in iterator sense (so actually one beyond
     the last element). Usually object.end().
   \return an iterator that points to the element in the sequence which 
     has the smallest (strictly) positive value. If no (strictly) positive
     element is found, or if the sequence is empty, the \a end argument
     is returned.

   The iterator type has to satisfy the requirements of a forward iterator,
   and its value_type has to be comparable using &lt; and &lt;=.
*/ 
template <typename ForwardIter_t>
ForwardIter_t 
min_positive_element(ForwardIter_t start, ForwardIter_t end)
{
  // go and look for the first (strictly) positive number
  while (start != end && *start <= 0) 
    ++start;
  if (start==end) return end;

  // now look through the rest for a smaller positive number
  ForwardIter_t result = start;
  while (++start != end)
    if(!(*start<=0) && *start<*result)
      result = start;

  return result;
}

END_NAMESPACE_STIR
#endif
