//
// $Id$
//
/*!
  \file
  \ingroup listmode
  \brief Implementation of class CListModeDataFromStream
    
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/


#include "local/stir/listmode/CListModeDataFromStream.h"
#include "local/stir/listmode/CListRecord.h"
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
  scanner_ptr =scanner_ptr_v;

  if (is_null_ptr(stream_ptr))
    return;
  starting_stream_position = stream_ptr->tellg();
  if (!stream_ptr->good())
    error("CListModeDataFromStream: error in tellg()\n");

  num_chars_left_in_buffer = 0;
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
  fstream* s_ptr = new fstream;
  open_read_binary(*s_ptr, listmode_filename.c_str());
  stream_ptr = s_ptr;
  if (reset() == Succeeded::no)
    error("CListModeDataFromStream: error in reset() for filename %s\n",
	  listmode_filename.c_str());

  scanner_ptr = scanner_ptr_v;
  num_chars_left_in_buffer = 0;
}

Succeeded
CListModeDataFromStream::
get_next_record(CListRecord& record) const
{

  if (is_null_ptr(stream_ptr))
    return Succeeded::no;

#if 0
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

  
  if (stream_ptr->good())
    return Succeeded::yes;
  if (stream_ptr->eof())
    return Succeeded::no; 
  else
  { error("Error after reading from stream in get_next_record\n"); }
  /* Silly statement to satisfy VC++, but we never get here */
  return Succeeded::no;
  
#else
  // cache the records in a 8 MB buffer

  // TODO this might skip last record in file
  static char buffer[8683520];
  static char *current_position_in_buffer = 0;

  // first check if we need to refill the buffer
  if (num_chars_left_in_buffer == 0)
  {
    //cerr << "Reading from listmode file \n";
    // read some more data
    const unsigned int buf_size = 8683520/size_of_record;
    stream_ptr->read(buffer, buf_size);
    current_position_in_buffer = buffer;
    num_chars_left_in_buffer = static_cast<unsigned int>(stream_ptr->gcount());
    if (stream_ptr->eof())
    {
    }
    else
    {
      if (!stream_ptr->good())
      { error("Error after reading from stream in CListModeDataFromStream::get_next_record\n"); }
      assert(buf_size==num_chars_left_in_buffer);
    }
  }

  // now get record from buffer
  if (num_chars_left_in_buffer!=0)
  {
    // next line will not work if CListRecord is a base-class
    //record = buffer[buffer_position++];
    memcpy(record.get_data_ptr(), current_position_in_buffer, size_of_record);
    current_position_in_buffer += size_of_record;
    num_chars_left_in_buffer-= size_of_record;
    if (do_byte_swap)
      {
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

    return Succeeded::yes;
  }
  else
  {
    return Succeeded::no;
  }
#endif

}



Succeeded
CListModeDataFromStream::
reset()
{
  if (is_null_ptr(stream_ptr))
    return Succeeded::no;

  // make sure we forget about any contents in the buffer
  num_chars_left_in_buffer = 0;

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
  streampos pos = stream_ptr->tellg();
  pos -= num_chars_left_in_buffer;
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
  stream_ptr->seekg(saved_get_positions[pos]);
  num_chars_left_in_buffer = 0;
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

#if 0
unsigned long
CListModeDataFromStream::
get_num_records() const
{ 
  // Determine maximum number of records from file size 
  input.seekg(0, ios::end);
  const streampos end_stream_position = input.tellg();
  
  return 
    static_cast<unsigned long>((end_stream_position - starting_stream_position) 
			       / sizeof(CListRecord));
}

#endif
END_NAMESPACE_STIR
