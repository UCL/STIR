//
// $Id$
//
/*!
  \file
  \ingroup listmode
  \brief Declarations of classes stir::CListRecordUsingUnion.
    
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

#ifndef __stir_listmode_CListRecordUsingUnion_H__
#define __stir_listmode_CListRecordUsingUnion_H__


#include "stir/listmode/CListRecord.h"

START_NAMESPACE_STIR

class CListModeDataFromStream;

//! A base class for records in list mode formats where all event types have the same size
/*! \ingroup listmode
  Many list mode formats have event types (such as timing or coincidence 
  events) that are the same type. Typically such a record has then a few 
  bits that identify the type of the record. So, reading a record in the
  data stream consists of reading a fixed number of bits. The type of
  record can then be determined by checking the relevant type-bits.

  It is possible to encode records in such a list mode format using a class 
  derived from CListRecordUsingUnion. The advantage is that all such
  list mode formats can be read using CListModeDataFromStream.
  It is expected that the derived class
  contains a \c union for the different types. 
  CListModeDataFromStream::get_next_record() can then simply read a specified
  number of bits and store it at the address of that union.
    
  \see CListModeData for more info on list mode data. 
*/
class CListRecordUsingUnion 
: public CListRecord, public CListTime, public CListEvent
{
protected:
  friend class CListModeDataFromStream;
  //! \name access to the raw data
  //@{
  /*! \warning Use with care!
      These functions exists (only) for allowing CListModeDataFromStream to 
      read the data.
  */
  virtual char const * get_const_data_ptr() const = 0;
  virtual char * get_data_ptr() = 0;
  //@}

};


END_NAMESPACE_STIR

#endif
