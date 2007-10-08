//
// $Id$
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - $Date$, Hammersmith Imanet Ltd
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
/*!
  \file 
  \ingroup Array
  \brief implementation of stir::convert_range

  \author Kris Thielemans
  \author PARAPET project

  $Date$

  $Revision$

*/
#include "stir/NumericInfo.h"
#include "stir/round.h"
#include <algorithm>
#include "boost/iterator/iterator_traits.hpp"
#include "boost/limits.hpp"
START_NAMESPACE_STIR

// anonymous namespace for local functions
namespace 
{

  /* Declaration of auxiliary function is_negative() with the obvious
     implementation.
     However, we overload it for unsigned types to return always false.
     The compiler would do this automatically for us. However, many 
     compilers (including gcc) will warn when you do
     unsigned x=...;
     if (x<0)
     { 
       // never get here
     }
     This file relies on templated definitions of convert_array. So,
     if we use if-statements as above with templated code, we will get
     warnings when instantiating the code with unsigned types.

     Summary, instead of the above if, write
     T x;
     if (is_negative(x))
     { 
       // never get here if T is an unsigned type
     }
     and you won't get a warning message
  */

  template <class T> 
  inline bool is_negative(const T x)
  { return x<0; }

  inline bool is_negative(const unsigned char x)
  { return false; }

  inline bool is_negative(const unsigned short x)
  { return false; }

  inline bool is_negative(const unsigned int x)
  { return false; }

  inline bool is_negative(const unsigned long x)
  { return false; }

}

template <class InputIteratorT, class T2, class scaleT>
inline void
find_scale_factor(scaleT& scale_factor,
		  const InputIteratorT& begin, const InputIteratorT& end,
		  const NumericInfo<T2> info_for_out_type)
{
  typedef typename boost::iterator_value<InputIteratorT>::type T1;
  NumericInfo<T1> info1;

  if (info1.type_id() == info_for_out_type.type_id())
  {
    // TODO could use different scale factor in this case as well, but at the moment we don't)
    scale_factor = scaleT(1);
    return;
  }

  // find the scale factor to use when converting to the maximum range in T2
  const double data_in_max =
    *std::max_element(begin, end);
  double tmp_scale = 
    data_in_max /
    static_cast<double>(info_for_out_type.max_value());
  if (info_for_out_type.signed_type() && info1.signed_type())
  {
    const double data_in_min =
    *std::min_element(begin, end);
    tmp_scale = 
      std::max(tmp_scale, 
	       data_in_min /static_cast<double>(info_for_out_type.min_value()));
  }
  // use an extra factor of 1.01. Otherwise, rounding errors can
  // cause data_in.find_max() / scale_factor to be bigger than the 
  // max_value
  tmp_scale *= 1.01;
  
  if (scale_factor == 0 || tmp_scale > scale_factor)
  {
    // We need to convert to the maximum range in T2
    scale_factor = scaleT(tmp_scale);
  }
}


template <class OutputIteratorT, class InputIteratorT, class scaleT>
void
  convert_range(const OutputIteratorT& out_begin,
		scaleT& scale_factor,
		const InputIteratorT& in_begin, const InputIteratorT& in_end)
{
    typedef typename boost::iterator_value<OutputIteratorT>::type OutType;

    find_scale_factor(scale_factor, in_begin, in_end, NumericInfo<OutType>());
    if (scale_factor == 0)
    {
      // data_in contains only 0
      OutputIteratorT out_iter = out_begin;
      InputIteratorT in_iter = in_begin;
      for (;
	   in_iter != in_end;
	   ++in_iter, ++out_iter)
	{    
	  *out_iter = static_cast<OutType>(0);
	}
      return;
    }
    
    // do actual conversion
    OutputIteratorT out_iter = out_begin;
    InputIteratorT in_iter = in_begin;
    if (!std::numeric_limits<OutType>::is_integer)
     {
	for (;
	     in_iter != in_end;
	     ++in_iter, ++out_iter)
	  {
	      *out_iter = 
		static_cast<OutType>(*in_iter / scale_factor);
	  } 
      }
    else
      {
	for (;
	     in_iter != in_end;
	     ++in_iter, ++out_iter)
	  {    
	    // KT coded the checks on the data types in the loop.
	    // This is presumably slow, but all these conditionals can be 
	    // resolved at compile time, so a good compiler does the work for me.
	    if (!std::numeric_limits<OutType>::is_signed
		&& is_negative(*in_iter))
	      {
		// truncate negatives
		*out_iter = 0;
	      }
	    else
	      {
		// convert using rounding
		*out_iter = 
		  static_cast<OutType>(round(*in_iter / scale_factor));
	      }
	  } 
      }
}

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
// specialisation for equal Iterator types
// In fact, we could do better and test for boost::iterator_value<InIteratorT>::type
// etc, but that requires some template trickery
template <class IteratorT, class scaleT>
void
  convert_range(const IteratorT& out_begin,
		scaleT& scale_factor,
		const IteratorT& in_begin, const IteratorT& in_end)
{
  scale_factor = scaleT(1);
  std::copy(in_begin, in_end, out_begin);
}
#endif


END_NAMESPACE_STIR
