//
// $Id$
//
#ifndef __Coordinate2D_H__
#define __Coordinate2D_H__
/*!
  \file 
  \ingroup buildblock 
  \brief defines the Coordinate2D<coordT> class 

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
  \ingroup buildblock 
  \brief a templated class for 2-dimensional coordinates.

   The only new method is a constructor Coordinate2D<coordT>(c1,c2)

   \warning Indices run from 1 to 2. 

*/

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

END_NAMESPACE_STIR

#include "stir/Coordinate2D.inl"

#endif

