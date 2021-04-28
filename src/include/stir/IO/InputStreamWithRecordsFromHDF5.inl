/*!
  \file
  \ingroup IO
  \ingroup GE
  \brief Implementation of class stir::GE::RDF_HDF5::InputStreamWithRecordsFromHDF5
    
  \author Kris Thielemans
  \author Ottavia Bertolli
  \author Nikos Efthimiou
  \author Palak Wadhwa
*/
/*
    Copyright (C) 2003-2011, Hammersmith Imanet Ltd
    Copyright (C) 2012-2013, Kris Thielemans
    Copyright (C) 2018 University of Hull
    Copyright (C) 2018 University of Leeds
    Copyright (C) 2020-2021 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/


#include "stir/utilities.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"
#include "stir/shared_ptr.h"

#include <fstream>
#include <string.h>
START_NAMESPACE_STIR

namespace GE {
namespace RDF_HDF5 {

//template <class RecordT>
//InputStreamWithRecordsFromHDF5<RecordT>::
//InputStreamWithRecordsFromHDF5(const shared_ptr<H5::DataSet>& dataset_sptr_v,
//                       const std::size_t size_of_record_signature,
//                       const std::size_t max_size_of_record)
//  : dataset_sptr(dataset_sptr_v),
//    size_of_record_signature(size_of_record_signature),
//    max_size_of_record(max_size_of_record)
//{
//  assert(size_of_record_signature<=max_size_of_record);
//  if (is_null_ptr(dataset_sptr))
//    return;
//  starting_stream_position = 0;
//  current_offset = 0;
//  //if (!dataset_sptr->good())
//  //  error("InputStreamWithRecordsFromHDF5: error in tellg()\n");
//}

template <class RecordT>
        InputStreamWithRecordsFromHDF5<RecordT>::
        InputStreamWithRecordsFromHDF5(const std::string& filename,
                                       const std::size_t size_of_record_signature,
                                       const std::size_t max_size_of_record):
    m_filename(filename),
    size_of_record_signature(size_of_record_signature),
    max_size_of_record(max_size_of_record)
{
    assert(size_of_record_signature<=max_size_of_record);

    this->max_buffer_size =10000000;
    this->buffer.reset(new char[this->max_buffer_size]);

    set_up();
}

template <class RecordT>
Succeeded
InputStreamWithRecordsFromHDF5<RecordT>::
set_up()
{
    input_sptr.reset(new GEHDF5Wrapper(m_filename));
    data_sptr.reset(new char[this->max_size_of_record]);
    starting_stream_position = 0;
    current_offset = 0;

    input_sptr->initialise_listmode_data();
    m_list_size = input_sptr->get_dataset_size() - this->size_of_record_signature;

    this->buffer_size = 0;
    return Succeeded::yes;
}

template <class RecordT>
void
InputStreamWithRecordsFromHDF5<RecordT>::
fill_buffer(const std::streampos offset) const
{
  this->buffer_size =
    static_cast<std::size_t>(std::min(static_cast<uint64_t>(this->max_buffer_size),
                                      m_list_size - offset));
  input_sptr->read_list_data(buffer.get(), offset, hsize_t(this->buffer_size));
  this->start_of_buffer_offset =  offset;
}

template <class RecordT>
void
InputStreamWithRecordsFromHDF5<RecordT>::
read_data(char* output,const std::streampos offset, const hsize_t size) const
{
  if (this->buffer_size == 0 || offset < this->start_of_buffer_offset ||
      offset >= (this->start_of_buffer_offset + static_cast<std::streampos>(this->buffer_size)))
    this->fill_buffer(offset);

  // copy data from buffer to output
  const std::size_t offset_in_buffer = offset - this->start_of_buffer_offset;
  const hsize_t size_in_buffer = std::min(size, static_cast<hsize_t>(this->buffer_size - offset_in_buffer));

  memcpy(output, this->buffer.get() + offset_in_buffer, static_cast<std::size_t>(size_in_buffer));

  // check if there is anything else to read after the end of the buffer
  if (size_in_buffer < size)
    read_data(output + size_in_buffer,
              offset + static_cast<std::streampos>(size_in_buffer),
              size - size_in_buffer);
}

template <class RecordT>
Succeeded
InputStreamWithRecordsFromHDF5<RecordT>::
get_next_record(RecordT& record)
{
  try
    {
      if (current_offset >= static_cast<std::streampos>(m_list_size))
        return Succeeded::no;
      char* data_ptr = data_sptr.get();
      this->read_data(data_ptr, current_offset, hsize_t(this->size_of_record_signature));
      const std::size_t size_of_record =
        record.size_of_record_at_ptr(data_ptr, this->size_of_record_signature, false);

      assert(size_of_record <= this->max_size_of_record);
      // read more bytes if necessary
      auto remainder = size_of_record - this->size_of_record_signature;
      if (remainder > 0)
        this->read_data(data_ptr+this->size_of_record_signature,
                       current_offset+static_cast<std::streampos>(this->size_of_record_signature),
                       hsize_t(remainder));
      current_offset += size_of_record;
      return
        record.init_from_data_ptr(data_ptr, size_of_record,false);
    }
  catch (...)
    {
      return Succeeded::no;
    }
}



template <class RecordT>
Succeeded
InputStreamWithRecordsFromHDF5<RecordT>::
reset()
{
  if (is_null_ptr(input_sptr))
    return Succeeded::no;

  current_offset = 0;
  return Succeeded::yes;
}


template <class RecordT>
typename InputStreamWithRecordsFromHDF5<RecordT>::SavedPosition
InputStreamWithRecordsFromHDF5<RecordT>::
save_get_position() 
{
  assert(!is_null_ptr(input_sptr));
  // TODO should somehow check if tellg() worked and return an error if it didn't

  saved_get_positions.push_back(current_offset);
  return saved_get_positions.size()-1;
} 

template <class RecordT>
Succeeded
InputStreamWithRecordsFromHDF5<RecordT>::
set_get_position(const typename InputStreamWithRecordsFromHDF5<RecordT>::SavedPosition& pos)
{
  if (is_null_ptr(input_sptr))
    return Succeeded::no;

  assert(pos < saved_get_positions.size());
  current_offset = saved_get_positions[pos];

  return Succeeded::yes;
}

template <class RecordT>
std::vector<std::streampos> 
InputStreamWithRecordsFromHDF5<RecordT>::
get_saved_get_positions() const
{
  return saved_get_positions;
}

template <class RecordT>
void 
InputStreamWithRecordsFromHDF5<RecordT>::
set_saved_get_positions(const std::vector<std::streampos>& poss)
{
  saved_get_positions = poss;
}

} // namespace
}
END_NAMESPACE_STIR
