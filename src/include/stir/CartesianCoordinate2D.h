#ifndef __CartesianCoordinate2D_H__
#define __CartesianCoordinate2D_H__
//
// $Id$
//
/*!
  \file 
 
  \brief defines the CartesianCoordinate2D<coordT> class 

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


#include "stir/Coordinate2D.h"


START_NAMESPACE_STIR

/*!
  \ingroup buildblock
   \brief a templated class for 2-dimensional coordinates.

   It is derived from Coordinate2D<coordT>. The only new methods are
   y(),x(), corresponding resp. to 
   operator[](1), operator[](2)

   \warning The constructor uses the order CartesianCoordinate2D<coordT>(y,x)
*/

template <typename coordT>
class CartesianCoordinate2D : public Coordinate2D<coordT>
{
protected:
  typedef Coordinate2D<coordT> base_type;
  typedef typename base_type::base_type basebase_type;

public:
  inline CartesianCoordinate2D();
  inline CartesianCoordinate2D(const coordT&, const coordT&);
  inline CartesianCoordinate2D(const basebase_type& c);
  inline CartesianCoordinate2D& operator=(const basebase_type& c);

  inline coordT& y();
  inline coordT y() const;
  inline coordT& x();
  inline coordT x() const;

};

END_NAMESPACE_STIR

#include "stir/CartesianCoordinate2D.inl"

#endif

