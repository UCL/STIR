#ifndef __Coordinate2D_H__
#define __Coordinate2D_H__
//
// $Id$: $Date$
//
/*
   This file declares class Coordinate2D<coordT>: 
   a templated class for 2-dimensional coordinates.
   It is derived from BasicCoordinate<2, coordT>. 
   The only new method is a constructor Coordinate2D<coordT>(c1,c2)

   Warning : 
   - Indices run from 1 to 2 

   History:
   1.0 (25/01/2000)
     Kris Thielemans and Alexey Zverovich
*/

#include "BasicCoordinate.h"

START_NAMESPACE_TOMO

template <typename coordT>
class Coordinate2D : public BasicCoordinate<2, coordT>
{
protected:
  typedef BasicCoordinate<2, coordT> base_type;

public:
  inline Coordinate2D();
  inline Coordinate2D(const coordT&, const coordT&);
  inline Coordinate2D(const base_type& c);
  inline Coordinate2D& operator=(const base_type& c);
};

END_NAMESPACE_TOMO

#include "Coordinate2D.inl"

#endif

