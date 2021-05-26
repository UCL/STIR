//
//
/*!
  \file
  \ingroup listmode
  \brief Declarations of classes stir::CListRecord, and stir::CListEvent which
  are used for list mode data.


  \author Nikos Efthimiou
  \author Daniel Deidda
  \author Kris Thielemans

*/
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
    Copyright (C) 2016, University of Hull
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

#ifndef __stir_listmode_CListRecord_H__
#define __stir_listmode_CListRecord_H__

#include "ListRecord.h"

START_NAMESPACE_STIR
class Bin;
class ProjDataInfo;
class Succeeded;
template <typename coordT>
class CartesianCoordinate3D;
template <typename coordT>
class LORAs2Points;

//! Class for storing and using a coincidence event from a list mode file
/*! \ingroup listmode
    CListEvent is used to provide an interface to the actual events (i.e.
    detected counts) in the list mode stream.

    \todo this is still under development. Things to add are for instance
    energy windows and time-of-flight info. Also, get_bin() would need
    time info or so for rotating scanners.

    \see CListModeData for more info on list mode data.
*/
class CListEvent : public ListEvent {
public:
  //! Changes the event from prompt to delayed or vice versa
  /*! Default implementation just returns Succeeded::no. */
  virtual Succeeded set_prompt(const bool prompt = true);

  //! Returns true is the delta_time has been swapped.
  bool get_swapped() const { return swapped; }

  double get_delta_time() const { return delta_time; }

protected:
  //! The detection time difference, between the two photons.
  //! This will work for ROOT files, but not so sure about acquired data.
  double delta_time;

  //! Indicates if the detectors' order has been swapped.
  bool swapped;

}; /*-coincidence event*/

class CListRecord : public ListRecord {
public:
  //! Used in TOF reconstruction to get both the geometric and the timing
  //!  component of the event
  virtual void full_event(Bin&, const ProjDataInfo&) const {
    error("CListRecord::full_event() is implemented only for records which "
          "hold timing and spatial information.");
  }
};

class CListRecordWithGatingInput : public CListRecord {};

END_NAMESPACE_STIR

#endif
