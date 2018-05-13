//
//
/*!
  \file
  \ingroup listmode
  \brief Declarations of classes stir::CListTime which
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

#ifndef __stir_listmode_CListTime_H__
#define __stir_listmode_CListTime_H__

#include "stir/round.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

//! A class for storing and using a timing record from a listmode file
/*! \ingroup listmode
    CListTime is used to provide an interface to the 'timing' events
    in the list mode stream. Usually, the timing event also contains
    gating information. For rotating scanners, it could also contain
    angle info.

    \todo this is still under development. Things to add are angles
    or so for rotating scanners. Also, some info on the maximum
    (and actual?) number of gates would be useful.
    \see CListModeData for more info on list mode data.
*/
class CListTime
{
public:
  virtual ~CListTime() {}

  virtual unsigned long get_time_in_millisecs() const = 0;
  inline double get_time_in_secs() const
    { return get_time_in_millisecs()/1000.; }

  virtual Succeeded set_time_in_millisecs(const unsigned long time_in_millisecs) = 0;
  inline Succeeded set_time_in_secs(const double time_in_secs)
    {
      unsigned long time_in_millisecs;
      round_to(time_in_millisecs, time_in_secs/1000.);
      return set_time_in_millisecs(time_in_millisecs);
    }

};

END_NAMESPACE_STIR

#endif

