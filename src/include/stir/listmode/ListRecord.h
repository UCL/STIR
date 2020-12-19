///
//
/*!
  \file
  \ingroup listmode
  \brief Declarations of classes stir::ListRecord which
  is used for list mode data.

  \author Daniel Deidda
  \author Kris Thielemans

*/
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
    Copyright (C) 2019, National Physical Laboratory
    Copyright (C) 2019, University College of London
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

#ifndef __stir_listmode_ListRecord_H__
#define __stir_listmode_ListRecord_H__

#include "ListEvent.h"
#include "ListTime.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

//! A class for a general element of a list mode file
/*! \ingroup listmode
    This represents either a timing or coincidence event in a list mode
    data stream.

    Some scanners can have more types of records. For example,
    the Quad-HiDAC puts singles information in the
    list mode file. If you need that information,
    you will have to do casting to e.g. ListRecordQHiDAC.

    \see ListModeData for more info on list mode data.
*/
class ListRecord
{
public:

    virtual ~ListRecord(){}

  virtual bool is_time() const = 0;

  virtual bool is_event() const = 0;

  virtual ListEvent&  event() = 0;
  virtual const ListEvent&  event() const = 0;
  virtual ListTime&   time() = 0;
  virtual const ListTime&   time() const = 0;
};

END_NAMESPACE_STIR

#endif
