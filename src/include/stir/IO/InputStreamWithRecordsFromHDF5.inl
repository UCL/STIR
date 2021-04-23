/*!
  \file
  \ingroup IO
  \ingroup GE
  \brief Implementation of class stir::GE::RDF_HDF5::InputStreamWithRecordsFromHDF5
    
  \author Kris Thielemans
  \author Ottavia Bertolli
*/
/*
    Copyright (C) 2003-2011, Hammersmith Imanet Ltd
    Copyright (C) 2012-2013, Kris Thielemans
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/


#include "stir/utilities.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"
#include "stir/shared_ptr.h"

#include <fstream>

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
    starting_stream_position = 0;
    current_offset = 0;

    set_up();
}

template <class RecordT>
Succeeded
InputStreamWithRecordsFromHDF5<RecordT>::
set_up()
{
    input_sptr.reset(new GEHDF5Wrapper(m_filename));
    data_sptr.reset(new char[this->max_size_of_record]);

    input_sptr->initialise_listmode_data();
    m_list_size = input_sptr->get_dataset_size() - this->size_of_record_signature;

    return Succeeded::yes;
}

template <class RecordT>
Succeeded
InputStreamWithRecordsFromHDF5<RecordT>::
get_next_record(RecordT& record)
{

  if (current_offset > static_cast<std::streampos>(m_list_size))
      return Succeeded::no;
  char* data_ptr = data_sptr.get();
  input_sptr->read_list_data(data_ptr, current_offset, hsize_t(this->size_of_record_signature));
  const std::size_t size_of_record =
    record.size_of_record_at_ptr(data_ptr, this->size_of_record_signature, false);

  assert(size_of_record <= this->max_size_of_record);
  // read more bytes if necessary
  auto remainder = size_of_record - this->size_of_record_signature;
  if (remainder > 0)
    input_sptr->read_list_data(data_ptr, current_offset, hsize_t(remainder));

  return
    record.init_from_data_ptr(data_ptr, size_of_record,false);
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
