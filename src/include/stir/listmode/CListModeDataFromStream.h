//
// $Id$
//
/*!
  \file
  \ingroup listmode
  \brief Declaration of class CListModeDataFromStream
    
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_listmode_CListModeDataFromStream_H__
#define __stir_listmode_CListModeDataFromStream_H__

#include "stir/listmode/CListModeData.h"
#include "stir/listmode/CListRecord.h"
#include "stir/shared_ptr.h"
#include "stir/ByteOrder.h"

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

//! A class that reads the listmode data from a (presumably binary) stream 
/*! This class is really a helper class for implementing different types
    of CListModeData types. It is useful when all types of records (e.g.
    timing and detected-event) have the same size. In that case, all IO
    handling is completely generic and is implemented in this class.

    The current implementation uses a buffered approach for IO. A fixed number
    of records is read into a buffer which is then used for subsequent
    calls to get_next_record(). So, disk IO is only necessary when the 
    buffer is empty.

    \warning Although this is currently a derived from CListModeData, this
    is possibly not a good design choice. In the end, we really need only
    the buffered IO. For example, to get the current implementation to work, the
    constructors need arguments such as \a empty_record_sptr such that the
    relevant members of CListModeData can be implemented. This increases
    dependency on CListModeData to a (probably) unacceptable level.
    (e.g. when adding CListModeData::num_energy_windows() we would need
    another argument here).

    \todo Do not derive this class from CListModeData.

    \todo This class currently relies in the fact that
     vector<streampos>::size_type == SavedPosition
*/
class CListModeDataFromStream : public CListModeData
{
public:
  //! Constructor taking a stream
  /*! Data will be assumed to start at the current position reported by seekg().*/ 
  CListModeDataFromStream(const shared_ptr<istream>& stream_ptr,
                          const shared_ptr<Scanner>& scanner_ptr,
			  const bool value_of_has_delayeds,
			  const size_t size_of_record,
			  const shared_ptr <CListRecord>& empty_record_sptr,
			  const ByteOrder list_mode_file_format_byte_order);

  //! Constructor taking a filename
  /*! File will be opened in binary mode. ListMode data will be assumed to 
      start at \a start_of_data.
  */
  CListModeDataFromStream(const string& listmode_filename, 
                          const shared_ptr<Scanner>& scanner_ptr,
			  const bool value_of_has_delayeds,
			  const size_t size_of_record,
			  const shared_ptr <CListRecord>& empty_record_sptr,
			  const ByteOrder list_mode_file_format_byte_order,
                          const streampos start_of_data = 0);

  virtual 
    shared_ptr <CListRecord> get_empty_record_sptr() const
    { return empty_record_sptr; }

  //! ECAT listmode data stores delayed events as well (as opposed to prompts)
  virtual bool has_delayeds() const 
    { return value_of_has_delayeds; }

  virtual 
    Succeeded get_next_record(CListRecord& event) const;

  virtual 
    Succeeded reset();

    SavedPosition save_get_position();

    Succeeded set_get_position(const SavedPosition&);

  //! Function that enables the user to store the saved get_positions
  /*! Together with set_saved_get_positions(), this allows 
      reinstating the saved get_positions when 
      reopening the same list-mode stream.
  */
  vector<streampos> get_saved_get_positions() const;
  //! Function that sets the saved get_positions
  /*! Normally, the argument results from a call to 
      get_saved_get_positions() on the same list-mode stream.
      \warning There is no check if the argument actually makes sense
      for the current stream.
  */ 
  void set_saved_get_positions(const vector<streampos>& );

private:

  const string listmode_filename;
  shared_ptr<istream> stream_ptr;
  streampos starting_stream_position;
  vector<streampos> saved_get_positions;

  const bool value_of_has_delayeds;
  const size_t size_of_record;

  const shared_ptr <CListRecord> empty_record_sptr;
  const ByteOrder list_mode_file_format_byte_order;  

  const bool do_byte_swap;

  mutable unsigned int num_chars_left_in_buffer;
};

END_NAMESPACE_STIR

#endif
