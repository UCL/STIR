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

#include "ListRecord.h"
#include "ListRecordWithGatingInput.h"
#include "ListEvent.h"
#include "ListGatingInput.h"
#include "ListTime.h"
#include "stir/round.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR
class Bin;
class ProjDataInfo;
class Succeeded;
template <typename coordT> class CartesianCoordinate3D;
template <typename coordT> class LORAs2Points;

//! Class for storing and using a coincidence event from a list mode file
/*! \ingroup listmode
    CListEvent is used to provide an interface to the actual events (i.e.
    detected counts) in the list mode stream.

    \todo this is still under development. Things to add are for instance
    energy windows and time-of-flight info. Also, get_bin() would need
    time info or so for rotating scanners.

    \see CListModeData for more info on list mode data. 
*/
class CListEvent : virtual public ListEvent
{
public:
  virtual ~CListEvent() {}

  //! Checks if this is a prompt event or a delayed event
  /*! PET scanners generally have a facility to detect events in a 
      'delayed' coincidence window. This is used to estimate the
      number of accidental coincidences (or 'randoms').
  */
  virtual
    bool
    is_prompt() const = 0;

  //! Changes the event from prompt to delayed or vice versa
  /*! Default implementation just returns Succeeded::no. */
  virtual 
    Succeeded
    set_prompt(const bool prompt = true);

  //! Finds the bin coordinates of this event for some characteristics of the projection data
  /*! bin.get_bin_value() will be <=0 when the event corresponds to
      an LOR outside the range of the projection data.

      bin.get_bin_value() will be set to a negative value if no such bin
      can be found.

      Currently, bin.get_bin_value() might indicate some weight
      which can be used for normalisation. This is unlikely
      to remain the case in future versions.

      The default implementation uses get_LOR()
      and ProjDataInfo::get_bin(). However, a derived class
      can overload this with a more efficient implementation.

    \todo get_bin() might need time info or so for rotating scanners.
  */
  virtual
    void
    get_bin(Bin& bin, const ProjDataInfo&) const;


}; /*-coincidence event*/

class CListTime : public ListTime
{
public:
  virtual ~CListTime() {}
};

class CListGatingInput : public ListGatingInput
{
public:
};

class CListRecord : public ListRecord
{
public:
  virtual ~CListRecord() {}
//  virtual bool operator==(const CListRecord& e2) const =0;
//  bool operator!=(const CListRecord& e2) const { return !(*this == e2); }

};

class CListRecordWithGatingInput : public CListRecord
{};

END_NAMESPACE_STIR

#endif
