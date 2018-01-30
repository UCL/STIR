/*
    Copyright (C) 2004 - 2007-10-08, Hammersmith Imanet Ltd
    Copyright (C) 2013, University College London
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
  //! enum specifying if the patient is scanned with the head first in the gantry, or the feet.
  enum OrientationValue
  { head_in, feet_in, other_orientation, unknown_orientation };
  //! enum specifying on what side the patient is lying on
  /*! \a prone is face-downward, \a left is lying on the left side */
  enum RotationValue
  { supine, prone, right, left, other_rotation, unknown_rotation };

  //! enum using DICOM abreviations
  /*! See Dicom C.7.3.1.1.2 */
  enum PositionValue
  {
    HFS, //!< Head First-Supine
    HFP, //!< Head First-Prone 	
    HFDR, //!< Head First-Decubitus Right 	
    HFDL, //!< Head First-Decubitus Left
    FFDR, //!< Feet First-Decubitus Right 	
    FFDL, //!< Feet First-Decubitus Left
    FFP, //!< Feet First-Prone 	
    FFS, //!< Feet First-Supine 
    unknown_position
  };

  //! Default constructor (setting to unknown position and orientation)
  PatientPosition()
    : orientation(unknown_orientation), rotation(unknown_rotation)
    {
      assert(rotation >=0);
      assert(rotation<= unknown_rotation);
      assert(orientation >=0);
      assert(orientation<= unknown_orientation);
    }

 PatientPosition(OrientationValue orientation, RotationValue rotation)
    : orientation(orientation), rotation(rotation)
    {
      assert(rotation >=0);
      assert(rotation<= unknown_rotation);
      assert(orientation >=0);
      assert(orientation<= unknown_orientation);
    }

  explicit PatientPosition(PositionValue position);

  void
    set_rotation(const RotationValue rotation_v)
  { 
    assert(rotation_v >=0);
    assert(rotation_v<= unknown_rotation);
    rotation = rotation_v; 
  }
  RotationValue
    get_rotation() const
  { return rotation; }

  void
    set_orientation(const OrientationValue orientation_v)
  { 
    assert(orientation_v >=0);
    assert(orientation_v<= unknown_orientation);
    orientation = orientation_v; 
  }
  OrientationValue
    get_orientation() const
  { return orientation; }

  PositionValue
    get_position() const;

  const char * const get_position_as_string() const;	
 private:
  OrientationValue orientation;
  RotationValue rotation;
};

END_NAMESPACE_STIR
#endif
