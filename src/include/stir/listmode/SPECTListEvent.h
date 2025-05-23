///
//
/*!
  \file
  \ingroup listmode
  \brief Declarations of classe stir::SPECTListEvent which
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

#ifndef __stir_listmode_SPECTListEvent_H__
#define __stir_listmode_SPECTListEvent_H__

#include "stir/round.h"
#include "stir/Succeeded.h"
#include "stir/listmode/ListEvent.h"
#include "stir/Bin.h"
#include "stir/ProjDataInfo.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/LORCoordinates.h"

START_NAMESPACE_STIR

//! Class for storing and using gamma events from a SPECT List mode file
/*! \ingroup listmode
    SPECTListEvent is used to provide an interface to the actual events (i.e.
    detected counts) in the list mode stream.

    \todo this is still under development. Things to add are for instance
    energy windows and time-of-flight info. Also, get_bin() would need
    time info or so for rotating scanners.

    \see SPECTListModeData for more info on list mode data.
*/
class SPECTListEvent: public ListEvent
{
public:
    virtual
      bool
      is_prompt() const {return true;}

}; /*-gamma event*/

END_NAMESPACE_STIR

#endif
