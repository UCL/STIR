#ifndef __Coordinate3D_H__
#define __Coordinate3D_H__
//
// $Id$: $Date$
//
/*
   This file declares class Coordinate3D<coordT>: 
   a templated class for 3-dimensional coordinates.
   It is derived from BasicCoordinate<3, coordT>. 
   The only new method is a constructor Coordinate3D<coordT>(c1,c2,c3)

   Warning : 
   - Indices run from 1 to 3, 

   History:
   1.0 (25/01/2000)
     Kris Thielemans and Alexey Zverovich
*/

#include "BasicCoordinate.h"

START_NAMESPACE_TOMO

template <typename coordT>
class Coordinate3D : public BasicCoordinate<3, coordT>
{
protected:
  typedef BasicCoordinate<3, coordT> base_type;

public:
  inline Coordinate3D();
  inline Coordinate3D(const coordT&, const coordT&, const coordT&);
  inline Coordinate3D(const base_type& c);
  inline Coordinate3D& operator=(const base_type& c);
};

END_NAMESPACE_TOMO

#include "Coordinate3D.inl"

#endif

