//
// $Id$
//
/*!
  \file
  \ingroup buildblock  
  \brief Declaration of class CListModeData
    
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_listmode_CListModeDataFromStream_H__
#define __stir_listmode_CListModeDataFromStream_H__

#include "local/stir/listmode/CListModeData.h"
#include "stir/shared_ptr.h"

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
class CListModeDataFromStream : public CListModeData
{
public:
  //! Constructor taking a stream
  /*! Data will be assumed to start at the current position reported by seekg().*/ 
  CListModeDataFromStream(const shared_ptr<istream>& stream_ptr,
                          const shared_ptr<Scanner>& scanner_ptr);

  //! Constructor taking a filename
  /*! File will be opened in binary mode. ListMode data will be assumed to 
      start at \a start_of_data.
  */
  CListModeDataFromStream(const string& listmode_filename, 
                          const shared_ptr<Scanner>& scanner_ptr,
                          const streampos start_of_data = 0);

  virtual 
    Succeeded get_next_record(CListRecord& event) const;

  virtual 
    Succeeded reset();

  virtual
    SavedPosition save_get_position();

  virtual
    Succeeded set_get_position(const SavedPosition&);

  //! Provide access to the associated stream
  /*! \warning This should not be used to read/write to the stream.
      Repositioning the stream is allowed though.
      \todo This is only provided to be able to save positions even
      after deleting the CListModeDataFromStream object. Instead, we should
      provide a way to keep those saved positions alive.
   */
  istream * const
    get_stream_ptr() const;

protected:
  string listmode_filename;
  shared_ptr<istream> stream_ptr;
  streampos starting_stream_position;
private:
  vector<streampos> saved_get_positions;
};

END_NAMESPACE_STIR

#endif
