#ifndef __Coordinate3D_H__
#define __Coordinate3D_H__
//
// $Id$
//
/*!
  \file 
 
  \brief defines the Coordinate3D<coordT> class 

  \author Sanida Mustafovic 
  \author Kris Thielemans 
  \author PARAPET project

  $Date$

  $Revision$

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/


#include "stir/BasicCoordinate.h"

START_NAMESPACE_STIR

/*!
   \brief a templated class for 3-dimensional coordinates.

   The only new method is a constructor Coordinate3D<coordT>(c1,c2,c3)

   \warning Indices run from 1 to 3.

*/

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

END_NAMESPACE_STIR

#include "stir/Coordinate3D.inl"

#endif

