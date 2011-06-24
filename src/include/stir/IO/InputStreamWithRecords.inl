//
// $Id$
//
/*!
  \file
  \ingroup IO
  \brief Implementation of class stir::InputStreamWithRecords
    
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
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


#include "stir/utilities.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"
#include <fstream>

#ifndef STIR_NO_NAMESPACES
using std::fstream;
using std::streamsize;
using std::streampos;
#endif

START_NAMESPACE_STIR
template <class RecordT, class OptionsT>
InputStreamWithRecords<RecordT, OptionsT>::
InputStreamWithRecords(const shared_ptr<istream>& stream_ptr,
                       const std::size_t size_of_record_signature,
                       const std::size_t max_size_of_record, 
                       const OptionsT& options)
  : stream_ptr(stream_ptr),
    size_of_record_signature(size_of_record_signature),
    max_size_of_record(max_size_of_record),
    options(options)
{
  assert(size_of_record_signature<=max_size_of_record);
  if (is_null_ptr(stream_ptr))
    return;
  starting_stream_position = stream_ptr->tellg();
  if (!stream_ptr->good())
    error("InputStreamWithRecords: error in tellg()\n");
}

template <class RecordT, class OptionsT>
InputStreamWithRecords<RecordT, OptionsT>::
InputStreamWithRecords(const string& filename,
                       const std::size_t size_of_record_signature,
                       const std::size_t max_size_of_record,
                       const OptionsT& options, 
                       const streampos start_of_data)
  : filename(filename),
    starting_stream_position(start_of_data),
    size_of_record_signature(size_of_record_signature),
    max_size_of_record(max_size_of_record),
    options(options)
{
  assert(size_of_record_signature<=max_size_of_record);
  fstream* s_ptr = new fstream;
  open_read_binary(*s_ptr, filename.c_str());
  stream_ptr = s_ptr;
  if (reset() == Succeeded::no)
    error("InputStreamWithRecords: error in reset() for filename %s\n",
	  filename.c_str());
}

template <class RecordT, class OptionsT>
Succeeded
InputStreamWithRecords<RecordT, OptionsT>::
get_next_record(RecordT& record) const
{
  if (is_null_ptr(stream_ptr))
    return Succeeded::no;

  // rely on file caching by the C++ library or the OS

  char data_ptr[this->max_size_of_record];
  stream_ptr->read(data_ptr, this->size_of_record_signature);
  if (stream_ptr->gcount()<static_cast<streamsize>(this->size_of_record_signature))
    return Succeeded::no; 
  const std::size_t size_of_record = record.size_of_record_at_ptr(data_ptr, this->size_of_record_signature,options);
  if (size_of_record > this->size_of_record_signature)
    stream_ptr->read(data_ptr + this->size_of_record_signature,
                     size_of_record - this->size_of_record_signature);
  if (stream_ptr->eof())
    return Succeeded::no; 
  else if (stream_ptr->bad())
    { 
      warning("Error after reading from list mode stream in get_next_record");
      return Succeeded::no; 
    }
  return 
    record.init_from_data_ptr(data_ptr, size_of_record,options);
}



template <class RecordT, class OptionsT>
Succeeded
InputStreamWithRecords<RecordT, OptionsT>::
reset()
{
  if (is_null_ptr(stream_ptr))
    return Succeeded::no;

  // Strangely enough, once you read past EOF, even seekg(0) doesn't reset the eof flag
  if (stream_ptr->eof()) 
    stream_ptr->clear();
  stream_ptr->seekg(starting_stream_position, std::ios::beg);
  if (stream_ptr->bad())
    return Succeeded::no;
  else
    return Succeeded::yes;
}


template <class RecordT, class OptionsT>
typename InputStreamWithRecords<RecordT, OptionsT>::SavedPosition
InputStreamWithRecords<RecordT, OptionsT>::
save_get_position() 
{
  assert(!is_null_ptr(stream_ptr));
  // TODO should somehow check if tellg() worked and return an error if it didn't
  streampos pos;
  if (!stream_ptr->eof())
    {
      pos = stream_ptr->tellg();
      if (!stream_ptr->good())
	error("InputStreamWithRecords<RecordT, OptionsT>::save_get_position\n"
	      "Error after getting position in file");
    }
  else
    {
      // use -1 to signify eof 
      // (this is probably the behaviour of tellg anyway, but this way we're sure).
      pos = streampos(-1); 
    }
  saved_get_positions.push_back(pos);
  return saved_get_positions.size()-1;
} 

template <class RecordT, class OptionsT>
Succeeded
InputStreamWithRecords<RecordT, OptionsT>::
set_get_position(const InputStreamWithRecords<RecordT, OptionsT>::SavedPosition& pos)
{
  if (is_null_ptr(stream_ptr))
    return Succeeded::no;

  assert(pos < saved_get_positions.size());
  if (saved_get_positions[pos] == streampos(-1))
    stream_ptr->seekg(0, std::ios::end); // go to eof
  else
    stream_ptr->seekg(saved_get_positions[pos]);
    
  if (!stream_ptr->good())
    return Succeeded::no;
  else
    return Succeeded::yes;
}

template <class RecordT, class OptionsT>
vector<streampos> 
InputStreamWithRecords<RecordT, OptionsT>::
get_saved_get_positions() const
{
  return saved_get_positions;
}

template <class RecordT, class OptionsT>
void 
InputStreamWithRecords<RecordT, OptionsT>::
set_saved_get_positions(const vector<streampos>& poss)
{
  saved_get_positions = poss;
}

END_NAMESPACE_STIR
