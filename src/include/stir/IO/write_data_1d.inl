/*!
  \file 
  \ingroup Array_IO_detail 
  \brief Implementation of stir::write_data_1d() functions

  \author Kris Thielemans

*/
/*
    Copyright (C) 2004- 2009, Hammersmith Imanet Ltd
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
#include "stir/Array.h"
#include "stir/Succeeded.h"
#include "stir/ByteOrder.h"
#include <fstream>

START_NAMESPACE_STIR

namespace detail {

/***************** version for ostream *******************************/

template <class elemT>
inline Succeeded
write_data_1d(std::ostream& s, const Array<1, elemT>& data,
	   const ByteOrder byte_order,
	   const bool can_corrupt_data)
{
  if (!s || 
      (dynamic_cast<std::ofstream*>(&s)!=0 && !dynamic_cast<std::ofstream*>(&s)->is_open()) || 
      (dynamic_cast<std::fstream*>(&s)!=0 && !dynamic_cast<std::fstream*>(&s)->is_open()))
    { warning("write_data: error before writing to stream.\n"); return Succeeded::no; }
  
  // While writing, the array is potentially byte-swapped.
  // We catch exceptions to prevent problems with this.
  // Alternative and safe way: (but involves creating an extra copy of the data)
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
  bool writing_ok=true;
  try
  {
    s.write(reinterpret_cast<const char *>(data.get_const_data_ptr()), num_to_write);
  }
  catch(...)
  {
    writing_ok=false;
  }

  data.release_const_data_ptr();	    

  if (!can_corrupt_data && !byte_order.is_native_order())
  {
    Array<1,elemT>& data_ref =
      const_cast<Array<1,elemT>&>(data);
    for(int i=data.get_min_index(); i<=data.get_max_index(); ++i)
      ByteOrder::swap_order(data_ref[i]);
  }

  if (!writing_ok || !s)
  { warning("write_data: error after writing to stream.\n"); return Succeeded::no; }

  return Succeeded::yes;
}

/***************** version for FILE *******************************/
// largely a copy of above, but with calls to stdio function

template <class elemT>
inline Succeeded
write_data_1d(FILE* & fptr_ref, const Array<1, elemT>& data,
	   const ByteOrder byte_order,
	   const bool can_corrupt_data)
{
  FILE *fptr = fptr_ref;
  if (fptr==0|| ferror(fptr))
    { warning("write_data: error before writing to FILE.\n"); return Succeeded::no; }
  
  // While writing, the array is potentially byte-swapped.
  // We catch exceptions to prevent problems with this.
  // Alternative and safe way: (but involves creating an extra copy of the data)
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
  const std::size_t num_to_write =
    static_cast<std::size_t>(data.size());
  const std::size_t num_written =
    fwrite(reinterpret_cast<const char *>(data.get_const_data_ptr()), sizeof(elemT), num_to_write, fptr);
  
  data.release_const_data_ptr();	    

  if (!can_corrupt_data && !byte_order.is_native_order())
  {
    Array<1,elemT>& data_ref =
      const_cast<Array<1,elemT>&>(data);
    for(int i=data.get_min_index(); i<=data.get_max_index(); ++i)
      ByteOrder::swap_order(data_ref[i]);
  }

  if (num_written!=num_to_write || ferror(fptr))
  { warning("write_data: error after writing to FILE.\n"); return Succeeded::no; }

  return Succeeded::yes;
}


} // end of namespace detail
END_NAMESPACE_STIR
