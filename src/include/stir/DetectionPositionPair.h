//
// $Id$
//

/*!
  \file
  \ingroup buildblock
  \brief Declaration of class DetectionPositionPair
  \author Kris Thielemans
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2002- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#ifndef __stir_DetectionPositionPair_H__
#define __stir_DetectionPositionPair_H__

#include "stir/DetectionPosition.h"

START_NAMESPACE_STIR
/*!
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
                               const DetectionPosition<coordT>&);
  
  inline const DetectionPosition<coordT>& pos1() const;   
  inline const DetectionPosition<coordT>& pos2() const;
  inline DetectionPosition<coordT>& pos1();   
  inline DetectionPosition<coordT>& pos2();
  //! comparison operators
  inline bool operator==(const DetectionPositionPair&) const;
  inline bool operator!=(const DetectionPositionPair&) const;
  
private :
  DetectionPosition<coordT>  p1;
  DetectionPosition<coordT>  p2;   
};

END_NAMESPACE_STIR

#include "stir/DetectionPositionPair.inl"

#endif //__DetectionPositionPairPair_H__
