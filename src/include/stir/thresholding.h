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
#include "stir/common.h"

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

END_NAMESPACE_STIR

#endif

