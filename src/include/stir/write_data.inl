// $Id$
/*!
  \file 
  \ingroup Array 
  \brief Implementation of write_data() functions for writing Arrays to file

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
#include "stir/Array.h"
#include "stir/convert_array.h"
#include "stir/NumericType.h"
#include "stir/NumericInfo.h"
#include "stir/Succeeded.h"
#include "stir/detail/test_if_1d.h"
#include <typeinfo>

START_NAMESPACE_STIR

/* This is the function that does the actual writing to std::ostream.
  \internal
 */
template <class elemT>
inline Succeeded
write_data_1d(std::ostream& s, const Array<1, elemT>& data,
	      const ByteOrder byte_order,
	      const bool can_corrupt_data);

template <class OStreamT, int num_dimensions, class elemT, class OutputType, class ScaleT>
Succeeded 
write_data(OStreamT& s, const Array<num_dimensions,elemT>& data, 
	   NumericInfo<OutputType> output_type, 
	   ScaleT& scale_factor,
	   const ByteOrder byte_order,
	   const bool can_corrupt_data)
{
  find_scale_factor(scale_factor,
		    data, 
		    NumericInfo<OutputType>());
  return
    write_data_with_fixed_scale_factor(s, data, output_type, 
				       scale_factor, byte_order,
				       can_corrupt_data);
}

template <class OStreamT, int num_dimensions, class elemT>
Succeeded 
write_data(OStreamT& s, const Array<num_dimensions,elemT>& data, 
	   const ByteOrder byte_order,
	   const bool can_corrupt_data)
{
  return
    write_data_with_fixed_scale_factor(s, data, NumericInfo<elemT>(),
				       1.F, byte_order,
				       can_corrupt_data);
}
  
namespace detail
{
  /* Generic implementation of write_data_with_fixed_scale_factor(). 
     See test_if_1d.h for info why we do this this way.
  */  

  template <class OStreamT, int num_dimensions, class elemT, class OutputType, class ScaleT>
  Succeeded 
  write_data_with_fixed_scale_factor_help(is_not_1d,
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

  // specialisation for 1D case
  template <class OStreamT, class elemT, class OutputType, class ScaleT>
  Succeeded 
  write_data_with_fixed_scale_factor_help(is_1d,
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
	if (new_scale_factor!=scale_factor)
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

} // end of namespace detail

template <class OStreamT, int num_dimensions, class elemT, class OutputType, class ScaleT>
Succeeded 
write_data_with_fixed_scale_factor(OStreamT& s, const Array<num_dimensions,elemT>& data, 
				   NumericInfo<OutputType> output_type, 
				   const ScaleT scale_factor,
				   const ByteOrder byte_order,
				   const bool can_corrupt_data)
{
  return 
    detail::
    write_data_with_fixed_scale_factor_help(detail::test_if_1d<num_dimensions>(),
					    s, data,
					    output_type, 
					    scale_factor, byte_order,
					    can_corrupt_data);
}


template <class elemT>
Succeeded
write_data_1d(std::ostream& s, const Array<1, elemT>& data,
	   const ByteOrder byte_order,
	   const bool can_corrupt_data)
{
  if (!s)
    { warning("write_data: error before writing to stream.\n"); return Succeeded::no; }
  
  // TODO handling of byte-swapping is unsafe if s.write() can throw. Check!
  // While writing, the array is byte-swapped.
  // Safe way: (but involves creating an extra copy of the data)
  /*
  if (!byte_order.is_native_order())
  {
  Array<1, elemT> a_copy(data);
  for(int i=data.get_min_index(); i<=data.get_max_index(); i++)
  ByteOrder::swap_order(a_copy[i]);
  return write_data(s, a_copy, ByteOrder::native, true);
  }
  */
  if (!byte_order.is_native_order())
  {
    Array<1,elemT>& data_ref =
      const_cast<Array<1,elemT>&>(data);
    for(int i=data.get_min_index(); i<=data.get_max_index(); ++i)
      ByteOrder::swap_order(data_ref[i]);
  }
  
  // note: find num_to_write (using size()) outside of s.write() function call
  // otherwise Array::check_state() in size() might abort if
  // get_const_data_ptr() is called before size() (which is compiler dependent)
  const std::streamsize num_to_write =
    static_cast<std::streamsize>(data.size())* sizeof(elemT);
  s.write(reinterpret_cast<const char *>(data.get_const_data_ptr()), num_to_write);
  data.release_const_data_ptr();	    

  if (!s)
  { warning("write_data: error after writing to stream.\n"); return Succeeded::no; }

  if (!can_corrupt_data && !byte_order.is_native_order())
  {
    Array<1,elemT>& data_ref =
      const_cast<Array<1,elemT>&>(data);
    for(int i=data.get_min_index(); i<=data.get_max_index(); ++i)
      ByteOrder::swap_order(data_ref[i]);
  }

  return Succeeded::yes;
}

template <class OStreamT, int num_dimensions, class elemT, class ScaleT>
Succeeded 
write_data(OStreamT& s, 
	   const Array<num_dimensions,elemT>& data, 
	   NumericType type, ScaleT& scale,
	   const ByteOrder byte_order,
	   const bool can_corrupt_data) 
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
#define CASE(NUMERICTYPE)                                \
    case NUMERICTYPE :                                   \
      return                                             \
        write_data(s, data,				 \
		   NumericInfo<typename TypeForNumericType<NUMERICTYPE >::type>(), \
		   scale, byte_order, can_corrupt_data)

      // now list cases that we want
#if !defined(_MSC_VER) || _MSC_VER>=1300
      CASE(NumericType::SCHAR);
#endif
      CASE(NumericType::SHORT);
      CASE(NumericType::USHORT);
      CASE(NumericType::FLOAT);
#undef CASE
    default:
      warning("write_data : type not yet supported\n, at line %d in file %s",
	       __LINE__, __FILE__); 
      return Succeeded::no;
      
    }

}

END_NAMESPACE_STIR

