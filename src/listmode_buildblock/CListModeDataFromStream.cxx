//
// $Id$
//
/*!
  \file
  \ingroup listmode
  \brief Implementation of class stir::CListModeDataFromStream
    
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


#include "stir/listmode/CListModeDataFromStream.h"
#include "stir/listmode/CListRecordUsingUnion.h"
#include "stir/utilities.h"
#include "stir/Succeeded.h"
#include "stir/ByteOrder.h"
#include "stir/is_null_ptr.h"
#include <fstream>

#ifndef STIR_NO_NAMESPACES
using std::fstream;
using std::streamsize;
using std::streampos;
#endif

START_NAMESPACE_STIR
CListModeDataFromStream::
CListModeDataFromStream(const shared_ptr<istream>& stream_ptr,
                        const shared_ptr<Scanner>& scanner_ptr_v,
			const bool has_delayeds, 
			const size_t size_of_record, 
			const shared_ptr<CListRecord>& empty_record_sptr,
			const ByteOrder list_mode_file_format_byte_order)
  : stream_ptr(stream_ptr),
    value_of_has_delayeds(value_of_has_delayeds),
    size_of_record(size_of_record),
    empty_record_sptr(empty_record_sptr),
    list_mode_file_format_byte_order(list_mode_file_format_byte_order),
    do_byte_swap(
		 size_of_record>1 &&
		 list_mode_file_format_byte_order != ByteOrder::get_native_order())
{
  // this run-time check on the type of the record should somehow be avoided
  // (by passing a shared_ptr of the correct type).
  if (dynamic_cast<CListRecordUsingUnion const *>(empty_record_sptr.get()) == 0)
    error("CListModeDataFromStream can only handle CListRecordUsingUnion types\n");

  scanner_ptr =scanner_ptr_v;

  if (is_null_ptr(stream_ptr))
    return;
  starting_stream_position = stream_ptr->tellg();
  if (!stream_ptr->good())
    error("CListModeDataFromStream: error in tellg()\n");
}

CListModeDataFromStream::
CListModeDataFromStream(const string& listmode_filename,
                        const shared_ptr<Scanner>& scanner_ptr_v,
			const bool has_delayeds, 
			const size_t size_of_record, 
			const shared_ptr<CListRecord>& empty_record_sptr,
			const ByteOrder list_mode_file_format_byte_order,
			const streampos start_of_data)
  : listmode_filename(listmode_filename),
    starting_stream_position(start_of_data),
    value_of_has_delayeds(value_of_has_delayeds),
    size_of_record(size_of_record),
    empty_record_sptr(empty_record_sptr),
    list_mode_file_format_byte_order(list_mode_file_format_byte_order),
    do_byte_swap(
		 size_of_record>1 &&
		 list_mode_file_format_byte_order != ByteOrder::get_native_order())

{
  // this run-time check on the type of the record should somehow be avoided
  // (by passing a shared_ptr of the correct type).
  if (dynamic_cast<CListRecordUsingUnion const *>(empty_record_sptr.get()) == 0)
    error("CListModeDataFromStream can only handle CListRecordUsingUnion types\n");
  fstream* s_ptr = new fstream;
  open_read_binary(*s_ptr, listmode_filename.c_str());
  stream_ptr = s_ptr;
  if (reset() == Succeeded::no)
    error("CListModeDataFromStream: error in reset() for filename %s\n",
	  listmode_filename.c_str());

  scanner_ptr = scanner_ptr_v;
}

std::time_t 
CListModeDataFromStream::
get_scan_start_time_in_secs_since_1970() const
{
  error("CListModeDataFromStream::get_scan_start_time_in_secs_since_1970() should never be called");
  return std::time_t(-1);
}

std::string
CListModeDataFromStream::
get_name() const
{
  error("CListModeDataFromStream::get_name() should never be called");
  return "";
}


Succeeded
CListModeDataFromStream::
get_next_record(CListRecord& record_of_general_type) const
{
  assert(dynamic_cast<CListRecordUsingUnion const *>(empty_record_sptr.get()) != 0);

  CListRecordUsingUnion& record =
    static_cast<CListRecordUsingUnion&>(record_of_general_type);
  if (is_null_ptr(stream_ptr))
    return Succeeded::no;

  // simple implementation that reads the records one by one from the stream
  // rely on file caching by the C++ library or the OS

  stream_ptr->read(record.get_data_ptr(), size_of_record);
  // now byte-swap it if necessary
  if (do_byte_swap)
  {
    //  ByteOrder::swap_order(record);
   switch (size_of_record)
     {
     case 2: revert_region<2>::revert(reinterpret_cast<unsigned char*>(record.get_data_ptr())); break;
     case 4: revert_region<4>::revert(reinterpret_cast<unsigned char*>(record.get_data_ptr())); break;
     case 6: revert_region<6>::revert(reinterpret_cast<unsigned char*>(record.get_data_ptr())); break;
     case 8: revert_region<8>::revert(reinterpret_cast<unsigned char*>(record.get_data_ptr())); break;
     case 10: revert_region<10>::revert(reinterpret_cast<unsigned char*>(record.get_data_ptr())); break;
     default: error("CListModeDataFromStream needs an extra line for this size (%d) of record at line %d\n",
		    size_of_record, __LINE__);
     }
  }
  record.init_from_data_ptr();

  
  if (stream_ptr->good())
    return Succeeded::yes;
  if (stream_ptr->eof())
    return Succeeded::no; 
  else
  { error("Error after reading from stream in get_next_record\n"); }
  /* Silly statement to satisfy VC++, but we never get here */
  return Succeeded::no;  

}



Succeeded
CListModeDataFromStream::
reset()
{
  if (is_null_ptr(stream_ptr))
    return Succeeded::no;

  // Strangely enough, once you read past EOF, even seekg(0) doesn't reset the eof flag
  if (stream_ptr->eof()) 
    stream_ptr->clear();
  stream_ptr->seekg(starting_stream_position, ios::beg);
  if (stream_ptr->bad())
    return Succeeded::no;
  else
    return Succeeded::yes;
}


CListModeData::SavedPosition
CListModeDataFromStream::
save_get_position() 
{
  assert(!is_null_ptr(stream_ptr));
  // TODO should somehow check if tellg() worked and return an error if it didn't
  streampos pos;
  if (!stream_ptr->eof())
    {
      pos = stream_ptr->tellg();
      if (!stream_ptr->good())
	error("CListModeDataFromStream::save_get_position\n"
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

Succeeded
CListModeDataFromStream::
set_get_position(const CListModeDataFromStream::SavedPosition& pos)
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

vector<streampos> 
CListModeDataFromStream::
get_saved_get_positions() const
{
  return saved_get_positions;
}

void 
CListModeDataFromStream::
set_saved_get_positions(const vector<streampos>& poss)
{
  saved_get_positions = poss;
}

END_NAMESPACE_STIR
