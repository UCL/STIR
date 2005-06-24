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
  \brief implementation of convert_array

  \author Kris Thielemans
  \author Charalampos Tsoumpas (UCHAR instatiations)
  \author PARAPET project

  $Date$

  $Revision$

  This file contains explicit instantiations. If you experience
  linking problems with convert_array, you might need to instantiate
  your own types at the end of this file.

  \warning Compiling this file will probably gives lots of warnings
  on testing negativity of unsigned types. This is because of
  template instantiations and is perfectly harmless.
  Ignore these warnings.

  Currently this file contains 2 implementations. The normal one, and
  one using full iterators. The latter is far more simple, but not
  supported by all compilers. Also, it is somewhat slower at the moment.

*/
// Because of support for compilers which cannot do partial
// template specialisation, this file is terribly messy.
// Try to read only the 'modern' stuff.

#include "stir/convert_array.h"
#include "stir/NumericInfo.h"
#include "stir/Array.h"
#include <algorithm>
// for floor
#include <math.h>

START_NAMESPACE_STIR


template <int num_dimensions, class T1, class T2, class scaleT>
void
find_scale_factor(scaleT& scale_factor,
		  const Array<num_dimensions,T1>& data_in, 
		  const NumericInfo<T2> info2)
{
  NumericInfo<T1> info1;

  if (info1.type_id() == info2.type_id())
  {
    // TODO could use different scale factor in this case as well, but at the moment we don't)
    scale_factor = scaleT(1);
    return;
  }

  // find the scale factor to use when converting to the maximum range in T2
  double tmp_scale = 
    static_cast<double>(data_in.find_max()) /  
    static_cast<double>(info2.max_value());
  if (info2.signed_type() && info1.signed_type())
  {
    tmp_scale = 
      max(tmp_scale, 
          static_cast<double>(data_in.find_min()) / 
	  static_cast<double>(info2.min_value()));
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

//! A local helper class in terms of which convert_array is defined
/*! Reason for this class is that we use partial template 
    specialisation (when supported by the compiler), and this
    is only possible with classes.
*/

// first for num_dimension > 1
template <int num_dimensions, class T1, class T2, class scaleT>
class convert_array_auxiliary
{
public:
  static inline void
    convert_array(Array<num_dimensions,T2>& data_out, 
                  scaleT& scale_factor,
                  const Array<num_dimensions,T1>& data_in)
  {
    
    NumericInfo<T2> info2;
    
    find_scale_factor(scale_factor, data_in, info2);
    if (scale_factor == 0)
    {
      // data_in contains only 0
      data_out.fill(0);
    }
    else
    {
      // do actual conversion
      convert_array_fixed_scale_factor(data_out, scale_factor, data_in);
    }
  }

  static inline void
    convert_array_fixed_scale_factor(
       Array<num_dimensions,T2>& data_out, 
       const scaleT scale_factor,
       const Array<num_dimensions,T1>& data_in)
  {
    typename Array<num_dimensions,T2>::iterator out_iter = data_out.begin();
    typename Array<num_dimensions,T1>::const_iterator in_iter = data_in.begin();
    const typename Array<num_dimensions,T1>::const_iterator in_iter_end = data_in.end();
    for (;
         in_iter != in_iter_end;
         ++in_iter, ++out_iter)
    {
      convert_array_auxiliary<num_dimensions-1,T1,T2,scaleT>::
        convert_array_fixed_scale_factor(*out_iter, scale_factor, *in_iter);
    }
  }
};


#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

// now for num_dimension == 1
template <class T1, class T2, class scaleT>
class convert_array_auxiliary<1, T1, T2, scaleT>
{
public:
  static inline void
    convert_array(Array<1,T2>& data_out, 
                  scaleT& scale_factor,
                  const Array<1,T1>& data_in)
  {
    
    NumericInfo<T2> info2;
    
    find_scale_factor(scale_factor, data_in, info2);
    if (scale_factor == 0)
    {
      // data_in contains only 0
      data_out.fill(0);
    }
    else
    {
      // do actual conversion
      convert_array_fixed_scale_factor(data_out, scale_factor, data_in);
    }
  }

  static inline void
    convert_array_fixed_scale_factor(
       Array<1,T2>& data_out, 
       const scaleT scale_factor,
       const Array<1,T1>& data_in)
  {
    // KT: I coded the various checks on the data types in the loop.
    // This is presumably slow, but all these conditionals can be 
    // resolved at compile time, so a good compiler does the work for me.
    typename Array<1,T2>::iterator out_iter = data_out.begin();
    typename Array<1,T1>::const_iterator in_iter = data_in.begin();
    const typename Array<1,T1>::const_iterator in_iter_end = data_in.end();
    for (;
         in_iter != in_iter_end;
         ++in_iter, ++out_iter)
    {    
      if (NumericInfo<T1>().signed_type() && !NumericInfo<T2>().signed_type() && *in_iter < 0)
      {
	// truncate negatives
	*out_iter = 0;
      }
      else
	*out_iter = 
 	  NumericInfo<T2>().integer_type() ?
	    static_cast<T2>(floor(*in_iter / scale_factor + 0.5)) :
            static_cast<T2>(*in_iter / scale_factor);
    }
  }
};

// special case T1=T2, num_dimensions>1
template <int num_dimensions, class T1, class scaleT>
class convert_array_auxiliary<num_dimensions,T1,T1,scaleT>
{
public:
  static inline void   
    convert_array(Array<num_dimensions,T1>& data_out, 
                  scaleT& scale_factor,
                  const Array<num_dimensions,T1>& data_in)
  {
    scale_factor = 1;
    data_out= data_in;
  }
};

// special case T1=T2, num_dimensions==1
// (needs to be here to sort out overloading of templates)
template <class T1, class scaleT>
class convert_array_auxiliary<1,T1,T1,scaleT>
{
public:
  static inline void
    convert_array(Array<1,T1>& data_out, 
                  scaleT& scale_factor,
                  const Array<1,T1>& data_in)
  {
    scale_factor = 1;
    data_out= data_in;
  }
};

#else // BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

// we'll assume scaleT==float
#define scaleT float

#define T1equalT2
#define T1 float
#define T2 float
#include "convert_array1d.cxx"
#define T1 short
#define T2 short
#include "convert_array1d.cxx"
#define T1 unsigned short
#define T2 short
#include "convert_array1d.cxx"
#define T1 signed char
#define T2 short
#include "convert_array1d.cxx"

#undef T1equalT2

#define T1 float
#define T2 signed char
#include "convert_array1d.cxx"
#define T1 signed char
#define T2 float
#include "convert_array1d.cxx"
#define T1 short
#define T2 signed char
#include "convert_array1d.cxx"
#define T1 signed char
#define T2 short
#include "convert_array1d.cxx"
#define T1 unsigned short
#define T2 signed char
#include "convert_array1d.cxx"
#define T1 signed char
#define T2 unsigned short
#include "convert_array1d.cxx"

#define T1 float
#define T2 short
#include "convert_array1d.cxx"
#define T1 float
#define T2 unsigned short
#include "convert_array1d.cxx"
#define T1 short
#define T2 float
#include "convert_array1d.cxx"
#define T1 short
#define T2 unsigned short
#include "convert_array1d.cxx"
#define T1 unsigned short
#define T2 short
#include "convert_array1d.cxx"
#define T1 unsigned short
#define T2 float
#include "convert_array1d.cxx"

#undef scaleT

#endif // BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION


/***************************************************************
 Finally, the implementation of convert_array
***************************************************************/
#if 1
template <int num_dimensions, class T1, class T2, class scaleT>
Array<num_dimensions, T2>
convert_array(scaleT& scale_factor,
	      const Array<num_dimensions, T1>& data_in, 
	      const NumericInfo<T2> info2)
{
  Array<num_dimensions,T2> data_out(data_in.get_index_range());

  convert_array(data_out, scale_factor, data_in);
  return data_out;    
}
#endif
#if 1
template <int num_dimensions, class T1, class T2, class scaleT>
void 
convert_array(Array<num_dimensions, T2>& data_out,
	      scaleT& scale_factor,
	      const Array<num_dimensions, T1>& data_in)
{
  convert_array_auxiliary<num_dimensions, T1, T2, scaleT>::
    convert_array(data_out, scale_factor, data_in);   
}
#endif

#if defined(ARRAY_FULL)
//TODO specialise for T1==T2
#if 1
template <int num_dimensions, class T1, class T2, class scaleT>
Array<num_dimensions, T2>
convert_array_FULL(scaleT& scale_factor,
	      const Array<num_dimensions, T1>& data_in, 
	      const NumericInfo<T2> info2)
{
  Array<num_dimensions,T2> data_out(data_in.get_index_range());

  convert_array_FULL(data_out, scale_factor, data_in);
  return data_out;    
}
#endif
#if 1
template <int num_dimensions, class T1, class T2, class scaleT>
void 
convert_array_FULL(Array<num_dimensions, T2>& data_out,
	      scaleT& scale_factor,
	      const Array<num_dimensions, T1>& data_in)
{
    assert(data_in.get_index_range() == data_out.get_index_range());

    NumericInfo<T2> info2;
    
    find_scale_factor(scale_factor, data_in, info2);
    if (scale_factor == 0)
    {
      // data_in contains only 0
      data_out.fill(0);
      return;
    }
    
    // do actual conversion
    // KT: I coded the various checks on the data types in the loop.
    // This is presumably slow, but all these conditionals can be 
    // resolved at compile time, so a good compiler does the work for me.
    typename Array<num_dimensions,T2>::full_iterator out_iter = 
      data_out.begin_all();
    typename Array<num_dimensions,T1>::const_full_iterator in_iter = 
      data_in.begin_all();
    const typename Array<num_dimensions,T1>::const_full_iterator in_iter_end = 
      data_in.end_all();
    for (;
         in_iter != in_iter_end;
         ++in_iter, ++out_iter)
    {    
	   //TODO can remove check on <t1>.signed_type
      if (NumericInfo<T1>().signed_type() && !NumericInfo<T2>().signed_type() && *in_iter < 0)
      {
	// truncate negatives
	*out_iter = 0;
      }
      else
	*out_iter = 
 	  NumericInfo<T2>().integer_type() ?
	    static_cast<T2>(floor(*in_iter / scale_factor + 0.5)) :
            static_cast<T2>(*in_iter / scale_factor);
    } 
}
#endif
#endif
/***************************************************************
  Template instantiations :
   for num_dimensions=1,2,3
   T1,T2 : float, short, scaleT : float
   T1,T2 : float, unsigned short, scaleT : float

   T1,T2 : short, unsigned short, scaleT: float (only for linking)
   T1=T2 : short, unsigned short, float, scaleT: float
***************************************************************/

// KT 05/07/2000 added explicit instantiation of void convert_array() version
// It is needed for VC in Release mode (and others?), because it inlines the 
// the definition of the 2nd version.
#define INSTANTIATE(dim, type_in, type_out) \
   template \
   void convert_array<>(Array<dim,type_out>& data_out, \
	      float& scale_factor, \
	      const Array<dim,type_in>& data_in); \
   template \
   Array<dim,type_out> convert_array<>( \
			      float& scale_factor, \
		              const Array<dim,type_in>& data_in, \
			      const NumericInfo<type_out> info2); \
   template \
   void find_scale_factor<>(float& scale_factor, \
                            const Array<dim,type_in>& data_in, \
			    const NumericInfo<type_out> info_out);
INSTANTIATE(1, float, signed char);
INSTANTIATE(1, float, unsigned char);
INSTANTIATE(1, signed char, float);
INSTANTIATE(1, unsigned char, float);
INSTANTIATE(1, short, signed char);
INSTANTIATE(1, signed char, short);
INSTANTIATE(1, unsigned short, signed char);
INSTANTIATE(1, signed char, unsigned short);
INSTANTIATE(1, short, unsigned char);
INSTANTIATE(1, unsigned char, short);
INSTANTIATE(1, unsigned short, unsigned char);
INSTANTIATE(1, unsigned char, unsigned short);

INSTANTIATE(1, float, short);
INSTANTIATE(1, float, unsigned short);
INSTANTIATE(1, short, float);
INSTANTIATE(1, unsigned short, float);
INSTANTIATE(1, unsigned short, short);
INSTANTIATE(1, short, unsigned short);

INSTANTIATE(1, signed char, signed char);
INSTANTIATE(1, unsigned char, unsigned char);
INSTANTIATE(1, short, short);
INSTANTIATE(1, unsigned short, unsigned short);
INSTANTIATE(1, float, float);

INSTANTIATE(2, float, signed char);
INSTANTIATE(2, float, unsigned char);
INSTANTIATE(2, signed char, float);
INSTANTIATE(2, unsigned char, float);
INSTANTIATE(2, short, signed char);
INSTANTIATE(2, signed char, short);
INSTANTIATE(2, unsigned short, signed char);
INSTANTIATE(2, signed char, unsigned short);
INSTANTIATE(2, short, unsigned char);
INSTANTIATE(2, unsigned char, short);
INSTANTIATE(2, unsigned short, unsigned char);
INSTANTIATE(2, unsigned char, unsigned short);

INSTANTIATE(2, float, short);
INSTANTIATE(2, float, unsigned short);
INSTANTIATE(2, short, float);
INSTANTIATE(2, unsigned short, float);
INSTANTIATE(2, unsigned short, short);
INSTANTIATE(2, short, unsigned short);

INSTANTIATE(2, signed char, signed char);
INSTANTIATE(2, unsigned char, unsigned char);
INSTANTIATE(2, short, short);
INSTANTIATE(2, unsigned short, unsigned short);
INSTANTIATE(2, float, float);

INSTANTIATE(3, float, signed char);
INSTANTIATE(3, float, unsigned char);
INSTANTIATE(3, signed char, float);
INSTANTIATE(3, unsigned char, float);

INSTANTIATE(3, short, signed char);
INSTANTIATE(3, signed char, short);
INSTANTIATE(3, unsigned short, signed char);
INSTANTIATE(3, signed char, unsigned short);

INSTANTIATE(3, short, unsigned char);
INSTANTIATE(3, unsigned char, short);
INSTANTIATE(3, unsigned short, unsigned char);
INSTANTIATE(3, unsigned char, unsigned short);

INSTANTIATE(3, float, short);
INSTANTIATE(3, float, unsigned short);
INSTANTIATE(3, short, float);
INSTANTIATE(3, unsigned short, float);
INSTANTIATE(3, unsigned short, short);
INSTANTIATE(3, short, unsigned short);

INSTANTIATE(3, signed char, signed char);
INSTANTIATE(3, unsigned char, unsigned char);
INSTANTIATE(3, short, short);
INSTANTIATE(3, unsigned short, unsigned short);
INSTANTIATE(3, float, float);

INSTANTIATE(4, float, signed char);
INSTANTIATE(4, float, unsigned char);
INSTANTIATE(4, signed char, float);
INSTANTIATE(4, unsigned char, float);

INSTANTIATE(4, short, signed char);
INSTANTIATE(4, signed char, short);
INSTANTIATE(4, unsigned short, signed char);
INSTANTIATE(4, signed char, unsigned short);

INSTANTIATE(4, short, unsigned char);
INSTANTIATE(4, unsigned char, short);
INSTANTIATE(4, unsigned short, unsigned char);
INSTANTIATE(4, unsigned char, unsigned short);

INSTANTIATE(4, float, short);
INSTANTIATE(4, float, unsigned short);
INSTANTIATE(4, short, float);
INSTANTIATE(4, unsigned short, float);
INSTANTIATE(4, unsigned short, short);
INSTANTIATE(4, short, unsigned short);

INSTANTIATE(4, unsigned char, unsigned char);
INSTANTIATE(4, signed char, signed char);
INSTANTIATE(4, short, short);
INSTANTIATE(4, unsigned short, unsigned short);
INSTANTIATE(4, float, float);

#undef INSTANTIATE

// TODO remove
#if defined(ARRAY_FULL)
#define INSTANTIATE(dim, type_in, type_out) \
   template \
   void convert_array_FULL<>(Array<dim,type_out>& data_out, \
	      float& scale_factor, \
	      const Array<dim,type_in>& data_in); \
   template \
   Array<dim,type_out> convert_array_FULL<>( \
			      float& scale_factor, \
		              const Array<dim,type_in>& data_in, \
			      const NumericInfo<type_out> info2); 

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
INSTANTIATE(1, float, signed char);
INSTANTIATE(1, float, unsigned char);
INSTANTIATE(1, signed char, float);
INSTANTIATE(1, unsigned char, float);
INSTANTIATE(1, short, signed char);
INSTANTIATE(1, signed char, short);
INSTANTIATE(1, unsigned short, signed char);
INSTANTIATE(1, signed char, unsigned short);
INSTANTIATE(1, short, unsigned char);
INSTANTIATE(1, unsigned char, short);
INSTANTIATE(1, unsigned short, unsigned char);
INSTANTIATE(1, unsigned char, unsigned short);
#endif
INSTANTIATE(1, float, short);
INSTANTIATE(1, float, unsigned short);
INSTANTIATE(1, short, float);
INSTANTIATE(1, unsigned short, float);
INSTANTIATE(1, unsigned short, short);
INSTANTIATE(1, short, unsigned short);

INSTANTIATE(1, short, short);
INSTANTIATE(1, unsigned short, unsigned short);
INSTANTIATE(1, float, float);

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
INSTANTIATE(2, float, signed char);
INSTANTIATE(2, float, unsigned char);
INSTANTIATE(2, signed char, float);
INSTANTIATE(2, unsigned char, float);
INSTANTIATE(2, short, signed char);
INSTANTIATE(2, signed char, short);
INSTANTIATE(2, unsigned short, signed char);
INSTANTIATE(2, signed char, unsigned short);
INSTANTIATE(2, short, unsigned char);
INSTANTIATE(2, unsigned char, short);
INSTANTIATE(2, unsigned short, unsigned char);
INSTANTIATE(2, unsigned char, unsigned short);
#endif
INSTANTIATE(2, float, short);
INSTANTIATE(2, float, unsigned short);
INSTANTIATE(2, short, float);
INSTANTIATE(2, unsigned short, float);
INSTANTIATE(2, unsigned short, short);
INSTANTIATE(2, short, unsigned short);

INSTANTIATE(2, short, short);
INSTANTIATE(2, unsigned short, unsigned short);
INSTANTIATE(2, float, float);

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
INSTANTIATE(3, float, signed char);
INSTANTIATE(3, float, unsigned char);
INSTANTIATE(3, signed char, float);
INSTANTIATE(3, short, signed char);
INSTANTIATE(3, signed char, short);
INSTANTIATE(3, unsigned short, signed char);
INSTANTIATE(3, signed char, unsigned short);
INSTANTIATE(3, unsigned char, float);
INSTANTIATE(3, short, unsigned char);
INSTANTIATE(3, unsigned char, short);
INSTANTIATE(3, unsigned short, unsigned char);
INSTANTIATE(3, unsigned char, unsigned short);
#endif
INSTANTIATE(3, float, short);
INSTANTIATE(3, float, unsigned short);
INSTANTIATE(3, short, float);
INSTANTIATE(3, unsigned short, float);
INSTANTIATE(3, unsigned short, short);
INSTANTIATE(3, short, unsigned short);

INSTANTIATE(3, short, short);
INSTANTIATE(3, unsigned short, unsigned short);
INSTANTIATE(3, float, float);

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
INSTANTIATE(4, float, signed char);
INSTANTIATE(4, float, unsigned char);
INSTANTIATE(4, signed char, float);
INSTANTIATE(4, short, signed char);
INSTANTIATE(4, signed char, short);
INSTANTIATE(4, unsigned short, signed char);
INSTANTIATE(4, signed char, unsigned short);
INSTANTIATE(4, unsigned char, float);
INSTANTIATE(4, short, unsigned char);
INSTANTIATE(4, unsigned char, short);
INSTANTIATE(4, unsigned short, unsigned char);
INSTANTIATE(4, unsigned char, unsigned short);
INSTANTIATE(4, signed char, signed char);
#endif
INSTANTIATE(4, float, short);
INSTANTIATE(4, float, unsigned short);
INSTANTIATE(4, short, float);
INSTANTIATE(4, unsigned short, float);
INSTANTIATE(4, unsigned short, short);
INSTANTIATE(4, short, unsigned short);

INSTANTIATE(4, short, short);
INSTANTIATE(4, unsigned short, unsigned short);
INSTANTIATE(4, float, float);


#undef INSTANTIATE

#endif
END_NAMESPACE_STIR
