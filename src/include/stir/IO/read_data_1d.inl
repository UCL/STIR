// $Id$
/*!
  \file 
  \ingroup Array_IO_detail 
  \brief Implementation of stir::read_data_1d() functions 

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
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

/***************** version for istream *******************************/

template <class elemT>
Succeeded
read_data_1d(std::istream& s, Array<1, elemT>& data,
	   const ByteOrder byte_order)
{
  if (!s || 
    (dynamic_cast<std::ifstream*>(&s)!=0 && !dynamic_cast<std::ifstream*>(&s)->is_open()) || 
      (dynamic_cast<std::fstream*>(&s)!=0 && !dynamic_cast<std::fstream*>(&s)->is_open()))
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

/***************** version for FILE *******************************/
// largely a copy of above, but with calls to stdio function

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
    fread(reinterpret_cast<char *>(data.get_data_ptr()), sizeof(elemT), num_to_read, fptr);
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


} // end of namespace detail
END_NAMESPACE_STIR
