//
// $Id$
//
/*!

  \file
  \ingroup buildblock  
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
#include "local/stir/listmode/lm.h"
#include "stir/utilities.h"
#include "stir/Succeeded.h"
#include "stir/ByteOrder.h"
#include "stir/ByteOrderDefine.h"
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
                        const shared_ptr<Scanner>& scanner_ptr_v)
  : stream_ptr(stream_ptr)
{
  if (is_null_ptr(stream_ptr))
    return;
  starting_stream_position = stream_ptr->tellg();
  if (!stream_ptr->good())
    error("CListModeDataFromStream: error in tellg()\n");

  scanner_ptr = scanner_ptr_v;
}

CListModeDataFromStream::
CListModeDataFromStream(const string& listmode_filename,
                        const shared_ptr<Scanner>& scanner_ptr_v,
			const streampos start_of_data)
  : listmode_filename(listmode_filename),
  starting_stream_position(start_of_data)  
{
  fstream* s_ptr = new fstream;
  open_read_binary(*s_ptr, listmode_filename.c_str());
  stream_ptr = s_ptr;
  if (reset() == Succeeded::no)
    error("CListModeDataFromStream: error in reset() for filename %s\n",
	  listmode_filename.c_str());

  scanner_ptr = scanner_ptr_v;
}

Succeeded
CListModeDataFromStream::
get_next_record(CListRecord& event) const
{
#ifdef STIRNativeByteOrderIsBigEndian
  assert(ByteOrder::get_native_order() == ByteOrder::big_endian);
#else
  assert(ByteOrder::get_native_order() == ByteOrder::little_endian);
#endif

#if 0  
  stream_ptr->read(reinterpret_cast<char *>(&event), sizeof(event));
#if (defined(STIRNativeByteOrderIsBigEndian) && !defined(STIRListModeFileFormatIsBigEndian)) \
    || (defined(STIRNativeByteOrderIsLittleEndian) && defined(STIRListModeFileFormatIsBigEndian)) 
  ByteOrder::swap_order(event);
#endif
  
  if (stream_ptr->good())
    return Succeeded::yes;
  if (stream_ptr->eof())
    return Succeeded::no; 
  else
  { error("Error after reading from stream in get_next_record\n"); }
  /* Silly statement to satisfy VC++, but we never get here */
  return Succeeded::no;
  
#else
// this will skip last event in file
  
  const unsigned int buf_size = 100000;
  static CListRecord buffer[buf_size];
  static unsigned int current_pos = buf_size;
  static streamsize num_records_in_buffer = 0;
  static streampos stream_position  = 0;
  if (current_pos == buf_size || stream_position != stream_ptr->tellg())// check if user reset the stream position, if so, reinitialise buffer
  {
    //cerr << "Reading from listmode file \n";
    // read some more data
    stream_ptr->read(reinterpret_cast<char *>(buffer), sizeof(event)*buf_size);
    current_pos=0;
    if (stream_ptr->eof())
    {
      num_records_in_buffer = stream_ptr->gcount();
    }
    else
    {
      if (!stream_ptr->good())
      { error("Error after reading from stream in get_next_record\n"); }
      num_records_in_buffer = buf_size;
      assert(buf_size*sizeof(event)==stream_ptr->gcount());
    }
    stream_position = stream_ptr->tellg();
    
  }

  if (current_pos != static_cast<unsigned int>(num_records_in_buffer))
  {
    event = buffer[current_pos++];
#if (defined(STIRNativeByteOrderIsBigEndian) && !defined(STIRListModeFileFormatIsBigEndian)) \
    || (defined(STIRNativeByteOrderIsLittleEndian) && defined(STIRListModeFileFormatIsBigEndian)) 
    ByteOrder::swap_order(event);
#endif
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
  stream_ptr->seekg(starting_stream_position, ios::beg);
  if (stream_ptr->bad())
    { 
      error("Error after seeking to start of data in CListModeDataFromStream::reset()\n");
      if (stream_ptr->eof()) 
	{ 
	  // Strangely enough, once you read past EOF, even seekg(0) doesn't reset the eof flag
	  stream_ptr->clear();
	  if (stream_ptr->eof()) 
	    error("seekg forgot to reset EOF or the file is empty. Can't correct this. Exiting...\n");      
	}
    }
  return Succeeded::yes;
}

#if 0
unsigned long
CListModeDataFromStream::
get_num_records() const
{ 
  // Determine maximum number of events from file size 
  input.seekg(0, ios::end);
  const streampos end_stream_position = input.tellg();
  
  return 
    static_cast<unsigned long>((end_stream_position - starting_stream_position) 
			       / sizeof(CListRecord));
}

#endif
END_NAMESPACE_STIR
