#ifndef __Coordinate4D_H__
#define __Coordinate4D_H__
//
// $Id$: $Date$
//
/*
   This file declares class Coordinate4D<coordT>: 
   a templated class for 4-dimensional coordinates.
   It is derived from BasicCoordinate<4, coordT>. 
   The only new method is a constructor Coordinate4D<coordT>(c1,c2,c3,c4)

   Warning : 
   - Indices run from 1 to 4 

   History:
   1.0 (25/01/2000)
     Kris Thielemans and Alexey Zverovich
*/

#include "BasicCoordinate.h"

START_NAMESPACE_TOMO

template <typename coordT>
class Coordinate4D : public BasicCoordinate<4, coordT>
{
protected:
  typedef BasicCoordinate<4, coordT> base_type;

public:
  inline Coordinate4D();
  inline Coordinate4D(const coordT&, const coordT&, const coordT&, const coordT&);
  inline Coordinate4D(const base_type& c);
  inline Coordinate4D& operator=(const base_type& c);
};

END_NAMESPACE_TOMO

#include "Coordinate4D.inl"

#endif

