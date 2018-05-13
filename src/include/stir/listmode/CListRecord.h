//
//
/*!
  \file
  \ingroup listmode
  \brief Declarations of classes stir::CListRecord, stir::CListTime and stir::CListEvent which
  are used for list mode data.
    
  \author Kris Thielemans
      
*/
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
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

#ifndef __stir_listmode_CListRecord_H__
#define __stir_listmode_CListRecord_H__

#include "stir/listmode/CListEvent.h"
#include "stir/listmode/CListTime.h"

START_NAMESPACE_STIR

//! A class recording external input to the scanner (normally used for gating)
/*! For some scanners, the state of some external measurements can be recorded in the
   list file, such as ECG triggers etc. We currently assume that these take discrete values.

   If your scanner has more data available, you can provide it in the derived class.
*/
class CListGatingInput
{
public:
  virtual ~CListGatingInput() {}

  //! get gating-related info
  /*! Generally, gates are numbered from 0 to some maximum value.
   */
  virtual unsigned int get_gating() const = 0;

  virtual Succeeded set_gating(unsigned int) = 0;
};

//! A class for a general element of a list mode file
/*! \ingroup listmode
    This represents either a timing or coincidence event in a list mode
    data stream.

    Some scanners can have more types of records. For example,
    the Quad-HiDAC puts singles information in the
    list mode file. If you need that information,
    you will have to do casting to e.g. CListRecordQHiDAC.
    
    \see CListModeData for more info on list mode data. 
*/
class CListRecord
{
public:
  virtual ~CListRecord() {}

  virtual bool is_time() const = 0;

  virtual bool is_event() const = 0;

  virtual CListEvent&  event() = 0;
  virtual const CListEvent&  event() const = 0;
  virtual CListTime&   time() = 0;
  virtual const CListTime&   time() const = 0;

  virtual bool operator==(const CListRecord& e2) const = 0;
  bool operator!=(const CListRecord& e2) const { return !(*this == e2); }

};

class CListRecordWithGatingInput : public CListRecord
{
 public:
  virtual bool is_gating_input() const { return false; }
  virtual CListGatingInput&  gating_input() = 0; 
  virtual const CListGatingInput&  gating_input() const = 0; 
};

END_NAMESPACE_STIR

#endif
