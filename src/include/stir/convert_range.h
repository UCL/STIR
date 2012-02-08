//
// $Id$
//
/*
    Copyright (C) 2006- $Date$, Hammersmith Imanet Ltd
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
#ifndef __stir_convert_range_H__
#define __stir_convert_range_H__

/*!
  \file 
  \ingroup Array
 
  \brief This file declares the stir::convert_range and stir::find_scale_factor functions.

  \author Kris Thielemans

  $Date$
  $Revision$

*/

#include "stir/common.h"

START_NAMESPACE_STIR

template <class T> class NumericInfo;

/*!
  \ingroup Array
   \brief A function that finds a scale factor to use when converting data to a new type

   This function works with input data given as an iterator range.
   \see find_scale_factor(scale_factor,data_in,info_for_out_type)
*/
template <class InputIteratorT, class T2, class scaleT>
inline void
find_scale_factor(scaleT& scale_factor,
		  const InputIteratorT& begin, const InputIteratorT& end,
		  const NumericInfo<T2> info_for_out_type);

/*!
  \ingroup Array
  \brief Converts the data in the input range to the output range (with elements of different types) such that \c data_in == \c data_out * \c scale_factor

  Note order of arguments. Output-range occurs first (as standard in STIR).
  \par example 
  \code
      convert_range(data_out.begin_all(), scale_factor, 
                    data_in.begin_all(), data_in.end_all());
  \endcode

  \see convert_array(scale_factor, data_in, info2) for more info
*/

template <class OutputIteratorT, class InputIteratorT, class scaleT>
inline void
  convert_range(const OutputIteratorT& out_begin,
		scaleT& scale_factor,
		const InputIteratorT& in_begin, const InputIteratorT& in_end);


END_NAMESPACE_STIR

#include "stir/convert_range.inl"

#endif

