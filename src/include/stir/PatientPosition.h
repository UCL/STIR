//
// $Id$
//
/*!
  \file
  \ingroup buildblock

  \brief Declaration of class PatientPosition

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
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
