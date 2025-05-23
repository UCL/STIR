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

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_IO_InputStreamWithRecords_H__
#define __stir_IO_InputStreamWithRecords_H__

#include "stir/shared_ptr.h"
#include "stir/Succeeded.h"

#include <iostream>
#include <string>
#include <vector>

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
  typedef std::vector<std::streampos>::size_type SavedPosition;
  //! Constructor taking a stream
  /*! Data will be assumed to start at the current position reported by seekg().
      If reset() is used, it will go back to this starting position.*/ 
  inline
    InputStreamWithRecords(const shared_ptr<std::istream>& stream_ptr,
                           const std::size_t size_of_record_signature,
                           const std::size_t max_size_of_record, 
                           const OptionsT& options);

  //! Constructor taking a filename
  /*! File will be opened in binary mode. Data will be assumed to 
      start at \a start_of_data.
  */
  inline
    InputStreamWithRecords(const std::string& filename, 
			   const std::size_t size_of_record_signature,
			   const std::size_t max_size_of_record, 
			   const OptionsT& options,
			   const std::streampos start_of_data = 0);

  virtual ~InputStreamWithRecords() {}

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
    std::vector<std::streampos> get_saved_get_positions() const;
  //! Function that sets the saved get_positions
  /*! Normally, the argument results from a call to 
      get_saved_get_positions() on the same stream.
      \warning There is no check if the argument actually makes sense
      for the current stream.
  */ 
  inline
    void set_saved_get_positions(const std::vector<std::streampos>& );

  inline
  std::istream& get_stream(){return *this->stream_ptr;}

private:
  shared_ptr<std::istream> stream_ptr;
  const std::string filename;

  std::streampos starting_stream_position;
  std::vector<std::streampos> saved_get_positions;

  const std::size_t size_of_record_signature;
  const std::size_t max_size_of_record;

  const OptionsT options;
};

END_NAMESPACE_STIR

#include "stir/IO/InputStreamWithRecords.inl"

#endif
