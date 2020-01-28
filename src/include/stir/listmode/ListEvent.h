///
//
/*!
  \file
  \ingroup listmode
  \brief Declarations of classe stir::ListEvent which
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

#ifndef __stir_listmode_ListEvent_H__
#define __stir_listmode_ListEvent_H__

#include "stir/round.h"
#include "stir/Succeeded.h"
#include "stir/Bin.h"
#include "stir/ProjDataInfo.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/LORCoordinates.h"

START_NAMESPACE_STIR

//! Class for storing and using gamma events from a list mode file
/*! \ingroup listmode
    ListEvent is used to provide an interface to the actual events (i.e.
    detected counts) in the list mode stream.

    \todo this is still under development. Things to add are for instance
    energy windows and time-of-flight info. Also, get_bin() would need
    time info or so for rotating scanners.

    \see ListModeData for more info on list mode data.
*/
class ListEvent
{
public:
  virtual ~ListEvent() {}
 virtual bool is_prompt() const =0;// {return helper_is_prompt();}

  //! Finds the LOR between the coordinates where the detection took place
  /*! Obviously, these coordinates are only estimates which depend on the
      scanner hardware. For example, Depth-of-Interaction might not be
      taken into account. However, the intention is that this function returns
      'likely' positions (e.g. not the face of a crystal, but a point somewhere
      in the middle).

      Coordinates are in mm and in the standard STIR coordinate system
      used by ProjDataInfo etc (i.e. origin is in the centre of the scanner).

      \todo This function might need time info or so for rotating scanners.
  */
  virtual LORAs2Points<float>
    get_LOR() const = 0;

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

  //! This method checks if the template is valid for LmToProjData
  /*! Used before the actual processing of the data (see issue #61), before calling get_bin()
   *  Most scanners have listmode data that correspond to non arc-corrected data and
   *  this check avoids a crash when an unsupported template is used as input.
   */
  virtual
  bool
  is_valid_template(const ProjDataInfo&) const =0;

}; /*-coincidence event*/

END_NAMESPACE_STIR

#endif
