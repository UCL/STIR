// $Id$
/*!
  \file 
  \ingroup Array 
  \brief Implementation of read_data() functions for reading Arrays from file

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
#include <iostream>

START_NAMESPACE_STIR

template <class IStreamT, int num_dimensions, class elemT, class InputType, class ScaleT>
Succeeded 
read_data(IStreamT& s, Array<num_dimensions,elemT>& data, 
	   NumericInfo<InputType> input_type, 
	   ScaleT& scale_factor,
	   const ByteOrder byte_order)
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

namespace detail
{
  /* Generic implementation of read_data(). See test_if_1d.h for info why we do this.*/  
  template <class IStreamT, int num_dimensions, class elemT>
  Succeeded 
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
  Succeeded 
  read_data_help(is_1d,
		 IStreamT& s, Array<1,elemT>& data, 
		 const ByteOrder byte_order)
  {
    return
      read_data_1d(s, data, byte_order);
  }

} // end of namespace detail

template <class IStreamT, int num_dimensions, class elemT>
Succeeded 
read_data(IStreamT& s, Array<num_dimensions,elemT>& data, 
	  const ByteOrder byte_order)
{
  return 
    detail::read_data_help(detail::test_if_1d<num_dimensions>(),
		   s, data, byte_order);
}


template <class elemT>
Succeeded
read_data_1d(std::istream& s, Array<1, elemT>& data,
	   const ByteOrder byte_order)
{
  if (!s)
    { warning("read_data: error before reading from stream.\n"); return Succeeded::no; }

  // note: find num_to_read (using size()) outside of s.read() function call
  // otherwise Array::check_state() in size() might abort if
  // get_data_ptr() is called before size() (which is compiler dependent)
  const std::streamsize num_to_read =
    static_cast<std::streamsize>(data.size())* sizeof(elemT);
  s.read(reinterpret_cast<char *>(data.get_data_ptr()), num_to_read);
  data.release_data_ptr();

  if (!s)
  { warning("read_data: error after reading from stream.\n"); return Succeeded::no; }
	    
  if (!byte_order.is_native_order())
  {
    for(int i=data.get_min_index(); i<=data.get_max_index(); ++i)
      ByteOrder::swap_order(data[i]);
  }

  return Succeeded::yes;
}


template <class elemT>
Succeeded
read_data_1d(FILE* & fptr_ref, Array<1, elemT>& data,
	   const ByteOrder byte_order)
{
  FILE *fptr = fptr_ref;
  if (fptr==NULL || ferror(fptr))
    { warning("read_data: error before reading from FILE.\n"); return Succeeded::no; }

  // note: find num_to_read (using size()) outside of s.read() function call
  // otherwise Array::check_state() in size() might abort if
  // get_data_ptr() is called before size() (which is compiler dependent)
  const std::size_t num_to_read =
    static_cast<std::size_t>(data.size());
  const std::size_t num_read =
    read(reinterpret_cast<char *>(data.get_data_ptr()), sizeof(elemT), num_to_read, fptr);
  data.release_data_ptr();

  if (ferror(fptr) || num_to_read!=num_read)
  { warning("read_data: error after reading from FILE.\n"); return Succeeded::no; }
	    
  if (!byte_order.is_native_order())
  {
    for(int i=data.get_min_index(); i<=data.get_max_index(); ++i)
      ByteOrder::swap_order(data[i]);
  }

  return Succeeded::yes;
}


template <class IStreamT, int num_dimensions, class elemT, class ScaleT>
Succeeded 
read_data(IStreamT& s, 
	  Array<num_dimensions,elemT>& data, 
	  NumericType type, ScaleT& scale,
	  const ByteOrder byte_order) 
{
  switch(type.id)
    {
      // define macro what to do with a specific NumericType
#define CASE(NUMERICTYPE)                                \
    case NUMERICTYPE :                                   \
      return                                             \
        read_data(s, data,				 \
		   NumericInfo<typename TypeForNumericType<NUMERICTYPE >::type>(), \
		   scale, byte_order)

      // now list cases that we want
#if !defined(_MSC_VER) || _MSC_VER>=1300
      CASE(NumericType::SCHAR);
#endif
      CASE(NumericType::SHORT);
      CASE(NumericType::USHORT);
      CASE(NumericType::FLOAT);
#undef CASE
    default:
      warning("read_data : type not yet supported\n, at line %d in file %s",
	       __LINE__, __FILE__); 
      return Succeeded::no;
      
    }

}

END_NAMESPACE_STIR

