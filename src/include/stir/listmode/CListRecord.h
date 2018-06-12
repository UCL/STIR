//
//
/*!
  \file
  \ingroup listmode
  \brief Declarations of classes stir::CListRecord which
  is used for list mode data.
    
  \author Nikos Efthimiou
  \author Kris Thielemans
      
*/
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
    Copyright (C) 2018 University of Hull
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

//! A class for a general element of a list mode file
/*! \ingroup listmode
    This represents either a timing or coincidence event in a list mode
    data stream.

    Some scanners can have more types of records. For example,
    the Quad-HiDAC puts singles information in the
    list mode file. If you need that information,
    you will have to do casting to e.g. CListRecordQHiDAC.
    
    \see CListModeData for more info on list mode data. 

    \author Kris Thielemans
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

END_NAMESPACE_STIR

#endif
