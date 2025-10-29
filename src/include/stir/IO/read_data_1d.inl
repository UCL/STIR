/*!
  \file
  \ingroup Array_IO_detail
  \brief Implementation of stir::read_data_1d() functions

  \author Kris Thielemans

*/
/*
    Copyright (C) 2004- 2009, Hammersmith Imanet Ltd
    Copyright (C) 2024, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#include "stir/Array.h"
#include "stir/Succeeded.h"
#include "stir/ByteOrder.h"
#include "stir/warning.h"
#include <fstream>

START_NAMESPACE_STIR

namespace detail
{

/***************** version for istream *******************************/

template <int num_dimensions, class elemT>
Succeeded
read_data_1d(std::istream& s, ArrayType<num_dimensions, elemT>& data, const ByteOrder byte_order)
{
  if (!s || (dynamic_cast<std::ifstream*>(&s) != 0 && !dynamic_cast<std::ifstream*>(&s)->is_open())
      || (dynamic_cast<std::fstream*>(&s) != 0 && !dynamic_cast<std::fstream*>(&s)->is_open()))
    {
      warning("read_data: error before reading from stream.\n");
      return Succeeded::no;
    }

  // note: find num_to_read (using size()) outside of s.read() function call
  // otherwise Array::check_state() in size() might abort if
  // get_data_ptr() is called before size() (which is compiler dependent)
  const std::streamsize num_to_read = static_cast<std::streamsize>(data.size_all()) * sizeof(elemT);
  s.read(reinterpret_cast<char*>(data.get_full_data_ptr()), num_to_read);
  data.release_full_data_ptr();

  if (!s)
    {
      warning("read_data: error after reading from stream.\n");
      return Succeeded::no;
    }

  if (!byte_order.is_native_order())
    {
      for (auto iter = data.begin_all(); iter != data.end_all(); ++iter)
        ByteOrder::swap_order(*iter);
    }

  return Succeeded::yes;
}

/***************** version for FILE *******************************/
// largely a copy of above, but with calls to stdio function

template <int num_dimensions, class elemT>
Succeeded
read_data_1d(FILE*& fptr_ref, ArrayType<num_dimensions, elemT>& data, const ByteOrder byte_order)
{
  FILE* fptr = fptr_ref;
  if (fptr == NULL || ferror(fptr))
    {
      warning("read_data: error before reading from FILE.\n");
      return Succeeded::no;
    }

  // note: find num_to_read (using size()) outside of s.read() function call
  // otherwise Array::check_state() in size() might abort if
  // get_data_ptr() is called before size() (which is compiler dependent)
  const std::size_t num_to_read = static_cast<std::size_t>(data.size_all());
  const std::size_t num_read = fread(reinterpret_cast<char*>(data.get_full_data_ptr()), sizeof(elemT), num_to_read, fptr);
  data.release_full_data_ptr();

  if (ferror(fptr) || num_to_read != num_read)
    {
      warning("read_data: error after reading from FILE.\n");
      return Succeeded::no;
    }

  if (!byte_order.is_native_order())
    {
      for (auto iter = data.begin_all(); iter != data.end_all(); ++iter)
        ByteOrder::swap_order(*iter);
    }

  return Succeeded::yes;
}

} // end of namespace detail
END_NAMESPACE_STIR
