//
// $Id$
//
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
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
/*!
  \file
  \ingroup buildblock

  \brief Declaration of class stir::PatientPosition

  \author Kris Thielemans

  $Date$
  $Revision$
*/

#ifndef __stir_PatientPosition_H__
#define __stir_PatientPosition_H__

#include "stir/common.h"

START_NAMESPACE_STIR
/*! \ingroup buildblock
  Class for encoding patient position
*/
class PatientPosition
{
 public:
  enum OrientationValues
    { head_in, feet_in, other_orientation };
  enum RotationValues
    { supine, prone, other_rotation };

  PatientPosition()
    : orientation(head_in), rotation(supine)
    {}

  void
    set_rotation(const RotationValues rotation_v)
  { 
    assert(rotation_v >=0);
    assert(rotation_v<= other_rotation);
    rotation = rotation_v; 
  }
  RotationValues
    get_rotation() const
  { return rotation; }

  void
    set_orientation(const OrientationValues orientation_v)
  { 
    assert(orientation_v >=0);
    assert(orientation_v<= other_orientation);
    orientation = orientation_v; 
  }
  OrientationValues
    get_orientation() const
  { return orientation; }

 private:
  OrientationValues orientation;
  RotationValues rotation;
};

END_NAMESPACE_STIR
#endif
