//
// $Id$
//
#ifndef __stir_thresholding_H__
#define  __stir_thresholding_H__
/*!
  \file
  \ingroup buildblock

  \brief Declaration of functions that threshold sequences (specified 
  by iterators).

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#include "stir/min_positive_element.h"
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::fill;
#endif

START_NAMESPACE_STIR

//! Threshold a sequence from above and below
/*! \par Type requirements:
    
    <ul>
    <li>forw_iterT is a forward iterator
    <li>elemT must be assignable to *forw_iterT
    <li>bool operator>(*forw_iterT, elemT) must exist
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
/*! \see threshold_upper_lower for type requirements */
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
/*! \see threshold_upper_lower for type requirements */
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
  \code *min_positive_element()*small_number
  \end_code
  However, if all values are less than or equal to 0, they are 
  set to \a small_number.

   \param start start of the sequence. Usually object.begin().
   \param end end of the sequence in iterator sense (so actually one beyond
     the last element). Usually object.end().
   \param small_number see above

   The iterator type has to satisfy the requirements of a forward iterator,
   and its value_type has to be comparable using < and <=.
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
    fill(begin, end, small_number);
}

END_NAMESPACE_STIR

#endif

