//
//
/*
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
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
#ifndef __stir_thresholding_H__
#define  __stir_thresholding_H__
/*!
  \file
  \ingroup buildblock

  \brief Declaration of functions that threshold sequences (specified 
  by iterators).

  \author Kris Thielemans

*/

#include "stir/min_positive_element.h"
#include <algorithm>

START_NAMESPACE_STIR
/*!  \ingroup buildblock
   \name Functions for thresholding numbers and sequences
*/
//@{

//! Threshold a sequence from above and below
/*!
  \par Type requirements:
    
    <ul>
    <li>\c forw_iterT is a forward iterator
    <li>\c elemT must be assignable to <tt>*forw_iterT</tt>
    <li><tt>bool operator&gt;(*forw_iterT, elemT)</tt> must exist
    </ul>
*/
template <typename forw_iterT, typename elemT>
inline void
threshold_upper_lower(forw_iterT begin, forw_iterT end,
		      const elemT new_min, const elemT new_max)
{
  for (forw_iterT iter = begin; iter != end; ++iter)
    {
      if (*iter > new_max)
	*iter = new_max;
      else
	if (new_min > *iter)
	  *iter = new_min;
    }
}

//! Threshold a sequence from above
/*! 
  \see threshold_upper_lower for type requirements */
template <typename forw_iterT, typename elemT>
inline void
threshold_upper(forw_iterT begin, forw_iterT end,
		const elemT new_max)
{
  for (forw_iterT iter = begin; iter != end; ++iter)
    {
      if (*iter > new_max)
	*iter = new_max;
    }
}

//! Threshold a sequence from below
/*! 
 \see threshold_upper_lower for type requirements */
template <typename forw_iterT, typename elemT>
inline void
threshold_lower(forw_iterT begin, forw_iterT end,
		const elemT new_min)
{
  for (forw_iterT iter = begin; iter != end; ++iter)
    {
      if (new_min > *iter)
	*iter = new_min;
    }
}

//! sets non-positive values in the sequence to small positive ones
/*!
  Thresholds the sequence from below to  
  <tt> *min_positive_element()*small_number</tt>.
  However, if all values are less than or equal to 0, they are 
  set to \a small_number.

   \param begin start of the sequence. Usually <tt>object.begin()</tt>.
   \param end end of the sequence in iterator sense (so actually one beyond
     the last element). Usually <tt>object.end()</tt>.
   \param small_number see above

   The iterator type has to satisfy the requirements of a forward iterator,
   and its value_type has to be comparable using &lt; and &lt;=.
*/ 
template <typename ForwardIter_t, typename elemT>
void
threshold_min_to_small_positive_value(ForwardIter_t begin, ForwardIter_t end, 
				      const elemT& small_number)
{
  const ForwardIter_t smallest_positive_element_iter =
    min_positive_element(begin, end);

  if (smallest_positive_element_iter!= end)
    threshold_lower(begin, end,  (*smallest_positive_element_iter)*small_number);
  else
    std::fill(begin, end, small_number);
}

//@}

END_NAMESPACE_STIR

#endif

