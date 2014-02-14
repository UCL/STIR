//
//
/*
    Copyright (C) 2002- 2007, Hammersmith Imanet Ltd
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
  \ingroup projdata
  \brief Declaration of class stir::DetectionPosition
  \author Kris Thielemans
*/
#ifndef __stir_DetectionPosition_H__
#define __stir_DetectionPosition_H__

#include "stir/common.h"

START_NAMESPACE_STIR
/*!   \ingroup projdata
 \brief
 A class for storing coordinates of a detection. 
 
 This encodes where
 a gamma photon was detected on a detector array. For example, in a
 cylindrical PET scanner, this class stores the crystal where a gamma
 was detected.
 
 The scanner might have more
 than 1 detector array (for example a dual-headed system), but this info is 
 (currently) not in this class. Also, the detector array might be rotating
 during acquisition. The corresponding angle is also not in the class.
 
 This is essentially a collection of 3 numbers: 
 <ol>
 <li>\c tangential_coord: a coordinate running tangentially to the 
 scanner cylinder, and orthogonal to the scanner axis. For a
 cylindrical PET scanner, this would be the Detection number in a ring.
 <li>\c axial_coord: a coordinate running along the scanner axis.
 For a cylindrical PET scanner, it would correspond to the ring number.
 <li>\c radial_coord: a coordinate 'orthogonal' to the 2 previous ones. This
 is only used for scanners with multiple layers of detectors (which would 
 give Depth Of Interaction information). radial_coord==0 corresponds to
 the layer closest to the centre of the scanner.
 </ol>
 All 3 coordinates are normally positive, and start with 0.
 \todo document directions
 
 For scanners that do not need 3 coordinates, there is a space overhead
 but no performance overhead, except in the comparison functions.
 The class is templated to allow for systems with continuous detection.
*/
template <typename coordT = unsigned int>
class DetectionPosition
{
public: 
  inline explicit
    DetectionPosition(const coordT tangential_coord=0,
  	                   const coordT axial_coord=0, 
			   const coordT radial_coord=0);
  
  inline coordT tangential_coord()  const;   
  inline coordT axial_coord()const;
  inline coordT radial_coord()const; 
  inline coordT& tangential_coord(); 
  inline coordT& axial_coord(); 
  inline coordT& radial_coord(); 
  //! \name comparison operators
  //@{
  inline bool operator==(const DetectionPosition&) const;
  inline bool operator!=(const DetectionPosition&) const;
  //@}
private :
  coordT  tangential;  
  coordT  axial; 
  coordT  radial;
};

END_NAMESPACE_STIR

#include "stir/DetectionPosition.inl"

#endif //__DetectionPosition_H__
