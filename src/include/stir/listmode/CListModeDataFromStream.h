//
// $Id$
//
/*!
  \file
  \ingroup listmode
  \brief Declaration of class stir::CListModeDataFromStream
    
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

//! A helper class to read listmode data from a (presumably binary) stream 
/*! \ingroup listmode
    This class is really a helper class for implementing different types
    of CListModeData types. It is useful when all types of records (e.g.
    timing and detected-event) have the same size. In that case, all IO
    handling is completely generic and is implemented in this class.

    \todo This class currently relies in the fact that
     vector<streampos>::size_type == SavedPosition
*/
class CListModeDataFromStream
{
public:
  typedef CListModeData::SavedPosition SavedPosition;
  //! Constructor taking a stream
  /*! Data will be assumed to start at the current position reported by seekg().*/ 
  CListModeDataFromStream(const shared_ptr<istream>& stream_ptr,
			  const size_t size_of_record,
			  const ByteOrder list_mode_file_format_byte_order);

  //! Constructor taking a filename
  /*! File will be opened in binary mode. ListMode data will be assumed to 
      start at \a start_of_data.
  */
  CListModeDataFromStream(const string& listmode_filename, 
			  const size_t size_of_record,
			  const ByteOrder list_mode_file_format_byte_order,
                          const streampos start_of_data = 0);

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

  const size_t size_of_record;

  const ByteOrder list_mode_file_format_byte_order;  

  const bool do_byte_swap;

};

END_NAMESPACE_STIR

#endif
