#ifndef __CartesianCoordinate3D_H__
#define __CartesianCoordinate3D_H__
//
// $Id$: $Date$
//
/*!
  \file 
  \ingroup buildblock 
  \brief defines the CartesianCoordinate3D<coordT> class 

  \author Sanida Mustafovic 
  \author Kris Thielemans 
  \author PARAPET project

  \date    $Date$
  \version $Revision$

*/


#include "Coordinate3D.h"
#ifdef OLDDESIGN
#include "pet_common.h"
#include "Point3D.h"
#endif

START_NAMESPACE_TOMO

/*!
  \ingroup buildblock
  \brief a templated class for 3-dimensional coordinates.

  It is derived from Coordinate3D<coordT>. The only new methods are
   z(),y(),x(), corresponding resp. to 
   operator[](1), operator[](2), operator[](3)

   \warning The constructor uses the order CartesianCoordinate3D<coordT>(z,y,x)
*/


template <typename coordT>
class CartesianCoordinate3D : public Coordinate3D<coordT>
{
protected:
  typedef Coordinate3D<coordT> base_type;
  typedef base_type::base_type basebase_type;

public:
  inline CartesianCoordinate3D();
  inline CartesianCoordinate3D(const coordT&, const coordT&, const coordT&);
  inline CartesianCoordinate3D(const basebase_type& c);
  inline CartesianCoordinate3D& operator=(const basebase_type& c);
#ifdef OLDDESIGN
  inline CartesianCoordinate3D(const Point3D& p);
#endif

  inline coordT& z();
  inline coordT z() const;
  inline coordT& y();
  inline coordT y() const;
  inline coordT& x();
  inline coordT x() const;

};

END_NAMESPACE_TOMO

#include "CartesianCoordinate3D.inl"

#endif

