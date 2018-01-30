/*
  Copyright (C) 2004 - 2008, Hammersmith Imanet Ltd
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
  \brief Implementation of stir::write_data() functions for writing stir::Array's to file

  \author Kris Thielemans

*/
#include "stir/Array.h"
#include "stir/convert_array.h"
#include "stir/NumericType.h"
#include "stir/NumericInfo.h"
#include "stir/ByteOrder.h"
#include "stir/Succeeded.h"
#include "stir/detail/test_if_1d.h"
#include "stir/IO/write_data_1d.h"
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

  __STIR_WORKAROUND_TEMPLATES==2
  The 2nd work-around no longer templates in num_dimensions. 
  For the least amount of pain, we also make the inlines here serve as 
  declarations, so use the first work-around for this.
*/

#ifndef __STIR_WORKAROUND_TEMPLATES
/* the normal case */

#  define ___BYTEORDER_DEFAULT
#  define ___CAN_CORRUPT_DATA_DEFAULT
#  define INT_NUM_DIMENSIONS int num_dimensions,

#else

#  define ___BYTEORDER_DEFAULT 	=ByteOrder::native
#  define ___CAN_CORRUPT_DATA_DEFAULT =false

#  if   __STIR_WORKAROUND_TEMPLATES==1

#    define INT_NUM_DIMENSIONS int num_dimensions,
   /* This case is for VC 6.0.
      Note that the order of the template arguments is important for VC 6.0.
      The "int num_dimensions" template argument has to be first in the
      template list, otherwise it chokes on it.
     */
#  else
/* horrible work-around for very deficient compilers
   We disable the num_dimensions template, and replace it by #defines.
*/

#    ifndef num_dimensions
#      error num_dimensions should be defined
#    endif

#    define INT_NUM_DIMENSIONS
#  endif
#endif

namespace detail
{
  /* Generic implementation of write_data_with_fixed_scale_factor(). 
     See test_if_1d.h for info why we do this this way.
  */  
#if !defined(__STIR_WORKAROUND_TEMPLATES) || __STIR_WORKAROUND_TEMPLATES<2 || num_dimensions!=1

  template < INT_NUM_DIMENSIONS class OStreamT,
	    class elemT, class OutputType, class ScaleT>
  inline Succeeded 
  write_data_with_fixed_scale_factor_help(
                                          is_not_1d,
					  OStreamT& s, const Array<num_dimensions,elemT>& data, 
					  NumericInfo<OutputType> output_type, 
					  const ScaleT scale_factor,
					  const ByteOrder byte_order,
					  const bool can_corrupt_data)
  {
    for (typename Array<num_dimensions,elemT>::const_iterator iter= data.begin();
	 iter != data.end();
	 ++iter)
      {
	if (write_data_with_fixed_scale_factor(s, *iter, output_type, 
					       scale_factor, byte_order,
					       can_corrupt_data) ==
	    Succeeded::no)
	  return Succeeded::no;
      }
    return Succeeded::yes;
  }
#endif

#if !defined(__STIR_WORKAROUND_TEMPLATES) || __STIR_WORKAROUND_TEMPLATES<2 || num_dimensions==1
  // specialisation for 1D case
  template <class OStreamT, class elemT, class OutputType, class ScaleT>
  inline Succeeded 
  write_data_with_fixed_scale_factor_help(
                                          is_1d,
					  OStreamT& s, const Array<1,elemT>& data, 
					  NumericInfo<OutputType>, 
					  const ScaleT scale_factor,
					  const ByteOrder byte_order,
					  const bool can_corrupt_data)
  {
    if (typeid(OutputType) != typeid(elemT) ||
	scale_factor!=1)
      {
	ScaleT new_scale_factor=scale_factor;
	Array<1,OutputType> data_tmp = 
	  convert_array(new_scale_factor, data, NumericInfo<OutputType>());
	if (std::fabs(new_scale_factor-scale_factor)> scale_factor*.001)
	  return Succeeded::no;
	return 
          write_data_1d(s, data_tmp, byte_order, /*can_corrupt_data*/ true);
      }
    else
      {
	return
	  write_data_1d(s, data, byte_order, can_corrupt_data);
      }
  }
#endif

} // end of namespace detail

template < INT_NUM_DIMENSIONS class OStreamT,
	  class elemT, class OutputType, class ScaleT>
Succeeded 
write_data_with_fixed_scale_factor(OStreamT& s, const Array<num_dimensions,elemT>& data, 
				   NumericInfo<OutputType> output_type, 
				   const ScaleT scale_factor,
				   const ByteOrder byte_order ___BYTEORDER_DEFAULT,
				   const bool can_corrupt_data ___CAN_CORRUPT_DATA_DEFAULT)
{
  return 
    detail::
    write_data_with_fixed_scale_factor_help(
                                            detail::test_if_1d<num_dimensions>(),
                                            s, data,
					    output_type, 
					    scale_factor, byte_order,
					    can_corrupt_data);
}

template < INT_NUM_DIMENSIONS class OStreamT,
	  class elemT, class OutputType, class ScaleT>
Succeeded 
write_data(OStreamT& s, const Array<num_dimensions,elemT>& data, 
	   NumericInfo<OutputType> output_type, 
	   ScaleT& scale_factor,
	   const ByteOrder byte_order ___BYTEORDER_DEFAULT,
	   const bool can_corrupt_data ___CAN_CORRUPT_DATA_DEFAULT)
{
  find_scale_factor(scale_factor,
		    data, 
		    NumericInfo<OutputType>());
  return
    write_data_with_fixed_scale_factor(s, data, output_type, 
				       scale_factor, byte_order,
				       can_corrupt_data);
}

template < INT_NUM_DIMENSIONS class OStreamT,
	  class elemT>
inline Succeeded 
write_data(OStreamT& s, const Array<num_dimensions,elemT>& data, 
	   const ByteOrder byte_order ___BYTEORDER_DEFAULT,
	   const bool can_corrupt_data ___CAN_CORRUPT_DATA_DEFAULT)
{
  return
    write_data_with_fixed_scale_factor(s, data, NumericInfo<elemT>(),
				       1.F, byte_order,
				       can_corrupt_data);
}

template < INT_NUM_DIMENSIONS class OStreamT,
	  class elemT, class ScaleT>
Succeeded 
write_data(OStreamT& s, 
	   const Array<num_dimensions,elemT>& data, 
	   NumericType type, ScaleT& scale,
	   const ByteOrder byte_order ___BYTEORDER_DEFAULT,
	   const bool can_corrupt_data ___CAN_CORRUPT_DATA_DEFAULT) 
{
  if (NumericInfo<elemT>().type_id() == type)
    {
      // you might want to use the scale even in this case, 
      // but at the moment we don't
      scale = ScaleT(1);
      return
	write_data(s, data, byte_order, can_corrupt_data);
    }
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
        write_data(s, data,				 \
		   NumericInfo<TYPENAME TypeForNumericType<NUMERICTYPE >::type>(), \
		   scale, byte_order, can_corrupt_data)

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
      warning("write_data : type not yet supported\n, at line %d in file %s",
	       __LINE__, __FILE__); 
      return Succeeded::no;
      
    }

}

#undef ___BYTEORDER_DEFAULT
#undef ___CAN_CORRUPT_DATA_DEFAULT
#undef INT_NUM_DIMENSIONS

END_NAMESPACE_STIR

