//
//
/*
    Copyright (C) 2005- 2005, Hammersmith Imanet Ltd
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
#ifndef __stir_more_algorithms_H__
#define __stir_more_algorithms_H__
/*!
  \file
  \ingroup buildblock
  
  \brief Declaration of some functions missing from std::algorithm
    
  \author Kris Thielemans

*/
#include "stir/common.h"
#include <iterator>

START_NAMESPACE_STIR

/*! \ingroup buildblock
  \brief Like std::max_element, but comparing after taking absolute value

  This function using stir::norm_squared(), so works for complex numbers as well.
*/
template <class iterT> 
inline
iterT abs_max_element(iterT start, iterT end);

/*!
  \ingroup buildblock
  \brief Compute the sum of a sequence using operator+=(), using an initial value.

  Sadly std::accumulate uses operator+(). For non-trivial objects,
  this might call an inefficient version for the addition. 
  As an alternative, std::accumulate could be called with a function object
  that calls operator+=.
*/
template <class IterT, class elemT>
inline 
elemT
sum(IterT start, IterT end, elemT init);

/*!
  \ingroup buildblock
  \brief Compute the sum of a sequence using operator+=().

  Sadly std::accumulate uses operator+(). For non-trivial objects,
  this might call an inefficient version for the addition. 
  Alternatively, std::accumulate could be called with a function object
  that calls operator+=. Still, in that case you need to specify an
  initial value. 

  If the range is empty, this function calls operator*=(0) to initialise the object to 0.

  \warning Currently, no work-around is present for old compilers that do not support 
  std::iterator_traits.
*/
template <class IterT>	
inline 
typename std::iterator_traits<IterT>::value_type
sum(IterT start, IterT end);

/*!
  \ingroup buildblock
  \brief Compute the average of a sequence using sum(start,end).
  \warning Currently, no work-around is present for old compilers that do not support 
  std::iterator_traits.
*/
template <class IterT>	
inline 
typename std::iterator_traits<IterT>::value_type
average(IterT start, IterT end);

END_NAMESPACE_STIR

#include "stir/more_algorithms.inl"
#endif
