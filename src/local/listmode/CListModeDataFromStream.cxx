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
  if (is_null_ptr(stream_ptr))
    return Succeeded::no;

#ifdef STIRNativeByteOrderIsBigEndian
  assert(ByteOrder::get_native_order() == ByteOrder::big_endian);
#else
  assert(ByteOrder::get_native_order() == ByteOrder::little_endian);
#endif

  // not sure what sizeof does with references. so check.
  assert(sizeof(event) == sizeof(CListRecord));
#if 1
  // simple implementation that reads the events one by one from the stream
  // rely on file caching by the C++ library or the OS

  /* On Windows XP with CYGWIN and gcc 3.2, rebinning a test file with this method
     gives the following times:
     real    0m17.952s
     user    0m7.811s
     sys     0m0.770s
     In contrast, the next ('cached by hand') version takes
     real    0m50.915s
     user    0m23.243s
     sys     0m11.165s
     This means, that the simplest version is by far faster (even though
     it requires almost continuous disk access). This is a due to the tellg() call apparently. See next version.
  */

  // next line will not work if CListRecord is a base-class (replace sizeof())
  stream_ptr->read(reinterpret_cast<char *>(&event), sizeof(event));
  // now byte-swap it if necessary
  // at the moment rely on STIRListModeFileFormatIsBigEndian preprocessor define
  // ugly! 
  // If this would be replaced by some conditional byte-swapping based on
  // a list_mode_file_format_byte_order member, this implementation would
  // be completely generic and could be used for arbitrary CListRecord 
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
  // cache the events in a 8 MB buffer

  // TODO this will skip last event in file
  // TODO this will currently break save_get_position() as the position in 
  // the buffer will not be taken into account
  const unsigned int buf_size = 8683520/sizeof(event);
    // next line will not work if CListRecord is a base-class
  static CListRecord buffer[buf_size];
  static unsigned int current_pos = buf_size;
  static streamsize num_records_in_buffer = 0;
  static streampos stream_position  = 0;

  // first check if we need to refill the buffer
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
      { error("Error after reading from stream in CListModeDataFromStream::get_next_record\n"); }
      num_records_in_buffer = buf_size;
      assert(buf_size*sizeof(event)==stream_ptr->gcount());
    }
    stream_position = stream_ptr->tellg();
    
  }

  // now get event from buffer
  if (current_pos != static_cast<unsigned int>(num_records_in_buffer))
  {
    // next line will not work if CListRecord is a base-class
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
  if (is_null_ptr(stream_ptr))
    return Succeeded::yes;

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
  saved_get_positions.push_back(stream_ptr->tellg());
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
  // Determine maximum number of events from file size 
  input.seekg(0, ios::end);
  const streampos end_stream_position = input.tellg();
  
  return 
    static_cast<unsigned long>((end_stream_position - starting_stream_position) 
			       / sizeof(CListRecord));
}

#endif
END_NAMESPACE_STIR
