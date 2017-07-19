//
//

/*!
  \file
  \ingroup projdata
  \brief Declaration of class stir::DetectionPositionPair
  \author Kris Thielemans
*/
/*
    Copyright (C) 2002- 2009, Hammersmith Imanet Ltd
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
#ifndef __stir_DetectionPositionPair_H__
#define __stir_DetectionPositionPair_H__

#include "stir/DetectionPosition.h"

START_NAMESPACE_STIR
/*!
  \ingroup projdata
  \brief
 A class for storing 2 coordinates-sets of a detection, as suitable for PET. 
 
 \see DetectionPosition for details on what we mean with a Detector Position
*/

template <typename coordT =  unsigned int>
class DetectionPositionPair
{
public: 
  inline DetectionPositionPair();
  
  inline DetectionPositionPair(const DetectionPosition<coordT>&, 
                               const DetectionPosition<coordT>&,
                               const coordT timing_pos = static_cast<coordT>(0));
  
  inline const DetectionPosition<coordT>& pos1() const;   
  inline const DetectionPosition<coordT>& pos2() const;
  inline const coordT timing_pos() const;
  inline DetectionPosition<coordT>& pos1();   
  inline DetectionPosition<coordT>& pos2();
  inline coordT& timing_pos();
  //! comparison operators
  inline bool operator==(const DetectionPositionPair&) const;
  inline bool operator!=(const DetectionPositionPair&) const;
  
private :
  DetectionPosition<coordT>  p1;
  DetectionPosition<coordT>  p2;   
  coordT _timing_pos;
};

END_NAMESPACE_STIR

#include "stir/DetectionPositionPair.inl"

#endif //__DetectionPositionPairPair_H__
