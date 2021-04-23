///
//
/*!
  \file
  \ingroup listmode
  \brief Declarations of classes stir::SPECTListRecord which
  is used for list mode data.

  \author Daniel Deidda
  \author Kris Thielemans

*/
/*
    Copyright (C) 2019, National Physical Laboratory
    Copyright (C) 2019, University College of London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_listmode_SPECTListRecord_H__
#define __stir_listmode_SPECTListRecord_H__


#include "stir/listmode/ListTime.h"
#include "ListRecord.h"
#include "stir/listmode/SPECTListEvent.h"

#include "stir/round.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

//! A class for a general element of a list mode file
/*! \ingroup listmode
    This represents either a timing or coincidence event in a list mode
    data stream.

    Some scanners can have more types of records.

    \see SPECTListModeData for more info on list mode data.
*/
class SPECTListRecord : public ListRecord
{
public:
//  virtual ~SPECTListRecord() {}

  virtual bool is_time() const = 0;

  virtual bool is_event() const = 0;

  virtual SPECTListEvent&  event() = 0;
  virtual const SPECTListEvent&  event() const = 0;
  virtual ListTime&   time() = 0;
  virtual const ListTime&   time() const = 0;

  virtual bool operator==(const SPECTListRecord& e2) const = 0;
  bool operator!=(const SPECTListRecord& e2) const { return !(*this == e2); }

};

END_NAMESPACE_STIR

#endif
