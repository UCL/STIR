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

#ifndef STIR_NO_NAMESPACES
using std::istream;
using std::string;
using std::streampos;
#endif

START_NAMESPACE_STIR

//! A class that reads the listmode data from a (presumably binary) stream 
class CListModeDataFromStream : public CListModeData
{
public:
  //! Constructor taking a stream
  /*! Data will be assumed to start at the current position reported by seekg().*/ 
  CListModeDataFromStream(const shared_ptr<istream>& stream_ptr);

  //! Constructor taking a filename
  /*! File will be opened in binary mode. ListMode data will be assumed to 
      start at \a start_of_data.
  */
  CListModeDataFromStream(const string& listmode_filename, 
                          const streampos start_of_data = 0);

  virtual 
    Succeeded get_next_record(CListRecord& event) const;

  virtual 
    Succeeded reset();

protected:
  string listmode_filename;
  shared_ptr<istream> stream_ptr;
  streampos starting_stream_position;
};

END_NAMESPACE_STIR

#endif
