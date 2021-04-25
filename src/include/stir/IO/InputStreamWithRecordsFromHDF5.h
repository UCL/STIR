//
//
/*!
  \file
  \ingroup IO
  \ingroup GE
  \brief Declaration of class stir::GE::RDF_HDF5::InputStreamWithRecordsFromHDF5
    
  \author Kris Thielemans
  \author Palak Wadhwa
  \author Ottavia Bertolli
  \author Nikos Efthimiou
      
*/
/*
    Copyright (C) 2016-2018, 2020-2021 University College London
    Copyright (C) 2016-2019, University of Leeds
    Copyright (C) 2016-2018, University of Hull

    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_IO_InputStreamWithRecordsFromHDF5_H__
#define __stir_IO_InputStreamWithRecordsFromHDF5_H__

#include "stir/shared_ptr.h"
#include "stir/Succeeded.h"
#include "stir/IO/GEHDF5Wrapper.h"
#include "boost/shared_array.hpp"
#include <string>
#include <iostream>
#include <vector>

START_NAMESPACE_STIR

namespace GE {
namespace RDF_HDF5 {

//! A helper class to read data from a hdf5 file
/*! \ingroup IO
    \ingroup GE

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
template <class RecordT>
class InputStreamWithRecordsFromHDF5
{
public:
  typedef std::vector<std::streampos>::size_type SavedPosition;
  //! Constructor taking a stream
  /*! Data will be assumed to start at the start of the DataSet.
      If reset() is used, it will go back to this starting position.*/ 
//  explicit
//    InputStreamWithRecordsFromHDF5(const shared_ptr<H5::DataSet>& ,
//                           const std::size_t size_of_record_signature,
//                           const std::size_t max_size_of_record);

  explicit
    InputStreamWithRecordsFromHDF5(const std::string& filename,
                           const std::size_t size_of_record_signature,
                           const std::size_t max_size_of_record);


  virtual ~InputStreamWithRecordsFromHDF5() {}

  inline
  virtual 
    Succeeded get_next_record(RecordT& record);

  virtual Succeeded set_up();

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

private:

  shared_ptr<GEHDF5Wrapper> input_sptr;

  boost::shared_array<char> data_sptr;

  uint64_t m_list_size = 0 ;

  std::streampos starting_stream_position;
  mutable std::streampos current_offset;
  std::vector<std::streampos> saved_get_positions;

  const std::string m_filename;
  const std::size_t size_of_record_signature;
  const std::size_t max_size_of_record;

  //! read data from buffer
  void read_data(char* output,const std::streampos offset, const hsize_t size) const;
  // members for buffering

  boost::shared_array<char> buffer;
  std::size_t max_buffer_size;
  //! currently filled size
  mutable std::size_t buffer_size;
  mutable std::streampos start_of_buffer_offset;
  void fill_buffer(const std::streampos offset) const;
};

} // namespace
}
END_NAMESPACE_STIR

#include "stir/IO/InputStreamWithRecordsFromHDF5.inl"

#endif
