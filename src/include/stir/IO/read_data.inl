/*
    Copyright (C) 2004- 2007, Hammersmith Imanet Ltd
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
  \ingroup Array_IO 
  \brief Implementation of stir::read_data() functions for reading stir::Array's from file

  \author Kris Thielemans

*/
#include "stir/Array.h"
#include "stir/convert_array.h"
#include "stir/NumericType.h"
#include "stir/NumericInfo.h"
#include "stir/Succeeded.h"
#include "stir/ByteOrder.h"
#include "stir/detail/test_if_1d.h"
#include "stir/IO/read_data_1d.h"
#include <typeinfo>

START_NAMESPACE_STIR

/* This file is made a bit more complicated because of various 
   work-arounds for older compilers.

  __STIR_WORKAROUND_TEMPLATES==1
  The first is probably specific for VC 6.0 where we need to define
  parameters for byte_order etc. in these definitions.
  In contrast, for the normal case, the defaults are in the .h file,
  and cannot be repeated here.
  I do this with some more preprocessor macros. Sigh.

  Note that write_data.inl has
  __STIR_WORKAROUND_TEMPLATES==2
  for even more broken compilers. I didn't bother to put this in here.
  You should be able to do this yourself (and then contribute the code).
*/

#ifndef __STIR_WORKAROUND_TEMPLATES
/* the normal case */

#  define ___BYTEORDER_DEFAULT

#else

#  define ___BYTEORDER_DEFAULT 	=ByteOrder::native

#endif 

namespace detail
{
  /* Generic implementation of read_data(). See test_if_1d.h for info why we do this.*/  
  template <int num_dimensions, class IStreamT, class elemT>
  inline Succeeded 
  read_data_help(is_not_1d,
		 IStreamT& s, Array<num_dimensions,elemT>& data, 
		 const ByteOrder byte_order)
  {
    for (typename Array<num_dimensions,elemT>::iterator iter= data.begin();
	 iter != data.end();
	 ++iter)
      {
	if (read_data(s, *iter, byte_order)==
	    Succeeded::no)
	  return Succeeded::no;
      }
    return Succeeded::yes;
  }

  /* 1D implementation of read_data(). See test_if_1d.h for info why we do this.*/  
  // specialisation for 1D case
  template <class IStreamT, class elemT>
  inline Succeeded 
  read_data_help(is_1d,
		 IStreamT& s, Array<1,elemT>& data, 
		 const ByteOrder byte_order)
  {
    return
      read_data_1d(s, data, byte_order);
  }

} // end of namespace detail

template <int num_dimensions, class IStreamT, class elemT>
inline Succeeded 
read_data(IStreamT& s, Array<num_dimensions,elemT>& data, 
	  const ByteOrder byte_order ___BYTEORDER_DEFAULT)
{
  return 
    detail::read_data_help(detail::test_if_1d<num_dimensions>(),
		   s, data, byte_order);
}


template <int num_dimensions, class IStreamT, class elemT, class InputType, class ScaleT>
inline Succeeded 
read_data(IStreamT& s, Array<num_dimensions,elemT>& data, 
	   NumericInfo<InputType> input_type, 
	   ScaleT& scale_factor,
	   const ByteOrder byte_order ___BYTEORDER_DEFAULT)
{
  if (typeid(InputType) == typeid(elemT))
    {
      // TODO? you might want to use the scale even in this case, 
      // but at the moment we don't
      scale_factor = ScaleT(1);
      return read_data(s, data, byte_order);
    }
  else
    {
      Array<num_dimensions,InputType> in_data(data.get_index_range());
      Succeeded success = read_data(s, in_data, byte_order);
      if (success == Succeeded::no)
	return Succeeded::no;
      convert_array(data, scale_factor, in_data);
      return Succeeded::yes;
    }
}

template <int num_dimensions, class IStreamT, class elemT, class ScaleT>
inline Succeeded 
read_data(IStreamT& s, 
	  Array<num_dimensions,elemT>& data, 
	  NumericType type, ScaleT& scale,
	  const ByteOrder byte_order ___BYTEORDER_DEFAULT) 
{
  switch(type.id)
    {
      // define macro what to do with a specific NumericType
#if !defined(_MSC_VER) || _MSC_VER>1300
#define TYPENAME typename
#else
#define TYPENAME 
#endif
#define CASE(NUMERICTYPE)                                \
    case NUMERICTYPE :                                   \
      return                                             \
        read_data(s, data,				 \
		   NumericInfo<TYPENAME TypeForNumericType<NUMERICTYPE >::type>(), \
		   scale, byte_order)

      // now list cases that we want
      CASE(NumericType::SCHAR);
      CASE(NumericType::UCHAR);
      CASE(NumericType::SHORT);
      CASE(NumericType::USHORT);
      CASE(NumericType::INT);
      CASE(NumericType::UINT);
      CASE(NumericType::LONG);
      CASE(NumericType::ULONG);
      CASE(NumericType::FLOAT);
      CASE(NumericType::DOUBLE);
#undef CASE
#undef TYPENAME
    default:
      warning("read_data : type not yet supported\n, at line %d in file %s",
	       __LINE__, __FILE__); 
      return Succeeded::no;
      
    }

}

END_NAMESPACE_STIR

