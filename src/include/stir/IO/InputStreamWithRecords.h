//
//
/*!
  \file
  \ingroup IO
  \brief Declaration of class stir::InputStreamWithRecords
    
  \author Kris Thielemans
      
*/
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
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

#ifndef __stir_IO_InputStreamWithRecords_H__
#define __stir_IO_InputStreamWithRecords_H__

#include "stir/shared_ptr.h"
#include "stir/Succeeded.h"

#include <iostream>
#include <string>
#include <vector>

#ifndef STIR_NO_NAMESPACES
using std::istream;
using std::string;
using std::streampos;
using std::vector;
#endif

START_NAMESPACE_STIR

//! A helper class to read data from a (presumably binary) stream 
/*! \ingroup IO
    This class is really a helper class for reading different records from a stream.
    It is useful when all types of records have some kind of signature to allow
    the function to find out what the size of the record is. In that case, all IO
    handling is completely generic and is implemented in this class.

    Current implementation needs a \c max_size_of_record to allocate enough memory 
    (on the stack) before reading. This is efficient in many cases, but impractical 
    in others.

    \par Requirements
    \c RecordT needs to have the following member functions
    \code
    std::size_t 
      size_of_record_at_ptr(const char * const buffer, const std::size_t size_available_in_buffer, 
                            const OptionsT options) const;

    Succeeded 
      init_from_data_ptr(const char * const buffer, 
                         const std::size_t size_of_record, 
                         const OptionsT options);
    \endcode

    \todo Allow choosing between allocation with \c new or on the stack.
*/
template <class RecordT, class OptionsT>
class InputStreamWithRecords
{
public:
  typedef vector<streampos>::size_type SavedPosition;
  //! Constructor taking a stream
  /*! Data will be assumed to start at the current position reported by seekg().
      If reset() is used, it will go back to this starting position.*/ 
  inline
    InputStreamWithRecords(const shared_ptr<istream>& stream_ptr,
                           const std::size_t size_of_record_signature,
                           const std::size_t max_size_of_record, 
                           const OptionsT& options);

  //! Constructor taking a filename
  /*! File will be opened in binary mode. Data will be assumed to 
      start at \a start_of_data.
  */
  inline
  InputStreamWithRecords(const string& filename, 
                          const std::size_t size_of_record_signature,
                          const std::size_t max_size_of_record, 
			  const OptionsT& options,
                          const streampos start_of_data = 0);

  inline
  virtual 
    Succeeded get_next_record(RecordT& record) const;

  //! go back to starting position
  inline
    Succeeded reset();

  //! save current "get" position in an internal array
  /*! \return an "index" into the array that allows you to go back.
      \see set_get_position
  */
  inline
  SavedPosition save_get_position();

  //! set current "get" position to previously saved value
  inline
  Succeeded set_get_position(const SavedPosition&);

  //! Function that enables the user to store the saved get_positions
  /*! Together with set_saved_get_positions(), this allows 
      reinstating the saved get_positions when 
      reopening the same stream.
  */
  inline
  vector<streampos> get_saved_get_positions() const;
  //! Function that sets the saved get_positions
  /*! Normally, the argument results from a call to 
      get_saved_get_positions() on the same stream.
      \warning There is no check if the argument actually makes sense
      for the current stream.
  */ 
  inline
  void set_saved_get_positions(const vector<streampos>& );

private:

  const string filename;
  shared_ptr<istream> stream_ptr;
  streampos starting_stream_position;
  vector<streampos> saved_get_positions;

  const std::size_t size_of_record_signature;
  const std::size_t max_size_of_record;

  const OptionsT options;
};

END_NAMESPACE_STIR

#include "stir/IO/InputStreamWithRecords.inl"

#endif
