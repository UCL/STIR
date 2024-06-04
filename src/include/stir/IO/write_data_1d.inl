/*!
  \file
  \ingroup Array_IO_detail
  \brief Implementation of stir::write_data_1d() functions

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

/***************** version for ostream *******************************/

template <int num_dimensions, class elemT>
inline Succeeded
write_data_1d(std::ostream& s, const Array<num_dimensions, elemT>& data, const ByteOrder byte_order, const bool can_corrupt_data)
{
  if (!s || (dynamic_cast<std::ofstream*>(&s) != 0 && !dynamic_cast<std::ofstream*>(&s)->is_open())
      || (dynamic_cast<std::fstream*>(&s) != 0 && !dynamic_cast<std::fstream*>(&s)->is_open()))
    {
      warning("write_data: error before writing to stream.\n");
      return Succeeded::no;
    }

  // While writing, the array is potentially byte-swapped.
  // We catch exceptions to prevent problems with this.
  // Alternative and safe way: (but involves creating an extra copy of the data)
  /*
  if (!byte_order.is_native_order())
  {
  Array<num_dimensions, elemT> a_copy(data);
  for(int i=data.get_min_index(); i<=data.get_max_index(); i++)
  ByteOrder::swap_order(a_copy[i]);
  return write_data(s, a_copy, ByteOrder::native, true);
  }
  */
  if (!byte_order.is_native_order())
    {
      Array<num_dimensions, elemT>& data_ref = const_cast<Array<num_dimensions, elemT>&>(data);
      for (auto iter = data_ref.begin_all(); iter != data_ref.end_all(); ++iter)
        ByteOrder::swap_order(*iter);
    }

  // note: find num_to_write (using size()) outside of s.write() function call
  // otherwise Array::check_state() in size() might abort if
  // get_const_data_ptr() is called before size() (which is compiler dependent)
  const std::streamsize num_to_write = static_cast<std::streamsize>(data.size_all()) * sizeof(elemT);
  bool writing_ok = true;
  try
    {
      s.write(reinterpret_cast<const char*>(data.get_const_full_data_ptr()), num_to_write);
    }
  catch (...)
    {
      writing_ok = false;
    }

  data.release_const_full_data_ptr();

  if (!can_corrupt_data && !byte_order.is_native_order())
    {
      Array<num_dimensions, elemT>& data_ref = const_cast<Array<num_dimensions, elemT>&>(data);
      for (auto iter = data_ref.begin_all(); iter != data_ref.end_all(); ++iter)
        ByteOrder::swap_order(*iter);
    }

  if (!writing_ok || !s)
    {
      warning("write_data: error after writing to stream.\n");
      return Succeeded::no;
    }

  return Succeeded::yes;
}

/***************** version for FILE *******************************/
// largely a copy of above, but with calls to stdio function

template <int num_dimensions, class elemT>
inline Succeeded
write_data_1d(FILE*& fptr_ref, const Array<num_dimensions, elemT>& data, const ByteOrder byte_order, const bool can_corrupt_data)
{
  FILE* fptr = fptr_ref;
  if (fptr == 0 || ferror(fptr))
    {
      warning("write_data: error before writing to FILE.\n");
      return Succeeded::no;
    }

  // While writing, the array is potentially byte-swapped.
  // We catch exceptions to prevent problems with this.
  // Alternative and safe way: (but involves creating an extra copy of the data)
  /*
  if (!byte_order.is_native_order())
  {
  Array<num_dimensions, elemT> a_copy(data);
  for(int i=data.get_min_index(); i<=data.get_max_index(); i++)
  ByteOrder::swap_order(a_copy[i]);
  return write_data(s, a_copy, ByteOrder::native, true);
  }
  */
  if (!byte_order.is_native_order())
    {
      Array<num_dimensions, elemT>& data_ref = const_cast<Array<num_dimensions, elemT>&>(data);
      for (auto iter = data_ref.begin_all(); iter != data_ref.end_all(); ++iter)
        ByteOrder::swap_order(*iter);
    }

  // note: find num_to_write (using size()) outside of s.write() function call
  // otherwise Array::check_state() in size() might abort if
  // get_const_data_ptr() is called before size() (which is compiler dependent)
  const std::size_t num_to_write = static_cast<std::size_t>(data.size_all());
  const std::size_t num_written
      = fwrite(reinterpret_cast<const char*>(data.get_const_full_data_ptr()), sizeof(elemT), num_to_write, fptr);

  data.release_const_full_data_ptr();

  if (!can_corrupt_data && !byte_order.is_native_order())
    {
      Array<num_dimensions, elemT>& data_ref = const_cast<Array<num_dimensions, elemT>&>(data);
      for (auto iter = data_ref.begin_all(); iter != data_ref.end_all(); ++iter)
        ByteOrder::swap_order(*iter);
    }

  if (num_written != num_to_write || ferror(fptr))
    {
      warning("write_data: error after writing to FILE.\n");
      return Succeeded::no;
    }

  return Succeeded::yes;
}

} // end of namespace detail
END_NAMESPACE_STIR
