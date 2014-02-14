//
//
/*
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
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
#ifndef __stir_phantoms_Utah_H__
#define __stir_phantoms_Utah_H__
/*!
  \file 
  \ingroup Shape
  \brief inline implementations for stir::Utah_phantom.

  \author Kris Thielemans
  \author Alexey Zverovich
  \author PARAPET project

*/

#include "stir/EllipsoidalCylinder.h"
#include "stir/CombinedShape3D.h"

START_NAMESPACE_STIR
  
/*!
  \brief A class that represents a Utah phantom.

  \todo dims here are wrong
  A: cylinder, 20cm diam, 10cm height
 B: cylinder, 18cm diam, 15cm height
 C: outer annulus, 20cm extern.diam, 
      2cm thick, 15cm height
 D: cylinder in B, 4.5cm diam, 18cm???
      height
 E: shorter cylinder in B, 4.5cm diam, 
      5.5cm height
  */
class Utah_phantom
{
public:
  /*! 
  \brief Construct a Utah phantom.

  It is oriented along z, with edge between A and B at z=0.
  */
  inline Utah_phantom();
  
  inline const shared_ptr<Shape3D>& get_A_ptr() const
  { return A_ptr; }
  
  inline const shared_ptr<Shape3D>& get_B_ptr() const
  { return B_ptr; }
  
  //! get B without holes
  inline const shared_ptr<Shape3D>& get_full_B_ptr() const
  { return full_B_ptr; }

  inline const shared_ptr<Shape3D>& get_C_ptr() const
  { return C_ptr; }
  
  //! get C without hole
  inline const shared_ptr<Shape3D>& get_full_C_ptr() const
  { return full_C_ptr; }

  inline const shared_ptr<Shape3D>& get_D_ptr() const
  { return D_ptr; }
  
  inline const shared_ptr<Shape3D>& get_E_ptr() const
  { return E_ptr; }

  /*! 
  \brief make a region inside B when \c fraction<1.

  The region has a smaller outer cylinder (scaled with fraction)
  and larger inner cylinders (scaled with 1/fraction).
  */
  inline shared_ptr<Shape3D> make_inside_B_ptr(const float fraction) const;

  /*! 
  \brief make a region inside C when \c fraction<1.

  The region has a smaller outer cylinder (scaled with fraction)
  and larger inner cylinder (scaled with 1/fraction).
  */
  inline shared_ptr<Shape3D> make_inside_C_ptr(const float fraction) const;

  inline void translate(const CartesianCoordinate3D<float>& direction);
  inline void scale(const CartesianCoordinate3D<float>& scale3D);

private:
  shared_ptr<Shape3D> A_ptr;
  shared_ptr<Shape3D> B_ptr;
  shared_ptr<Shape3D> full_B_ptr;
  shared_ptr<Shape3D> C_ptr;
  shared_ptr<Shape3D> full_C_ptr;
  shared_ptr<Shape3D> D_ptr;
  shared_ptr<Shape3D> E_ptr;
};

Utah_phantom::Utah_phantom()
{
//EllipsoidalCylinder (Lcyl,Rcyl_a,Rcyl_b,
//    CartesianCoordinate3D<float>(zc,yc,xc),
//    alpha,beta,gamma)


  A_ptr = new EllipsoidalCylinder (100,100,100,
    CartesianCoordinate3D<float>(-50,0,0));
  full_B_ptr = new EllipsoidalCylinder (150,80,80,
    CartesianCoordinate3D<float>(75,0,0));
  full_C_ptr = new EllipsoidalCylinder (150,100,100,
    CartesianCoordinate3D<float>(75,0,0));
  D_ptr = new EllipsoidalCylinder (105,22.5,22.5,
    CartesianCoordinate3D<float>(52.5,0,-47.5));
  E_ptr = new EllipsoidalCylinder (55,22.5,22.5,
    CartesianCoordinate3D<float>(27.5,0,47.5));
  
  //CombinedShape3D< logical_and_not<bool> > C( &FullC,&FullB);
  C_ptr = new CombinedShape3D< logical_and_not<bool> > ( full_C_ptr,full_B_ptr);
  shared_ptr<Shape3D> full_B_notD_ptr = 
    new CombinedShape3D< logical_and_not<bool> > ( full_B_ptr,D_ptr);
  B_ptr = new CombinedShape3D< logical_and_not<bool> > ( full_B_notD_ptr,E_ptr);
}

shared_ptr<Shape3D> 
Utah_phantom::make_inside_B_ptr(const float fraction) const
{
  // first get a new copy of full_B
  shared_ptr<Shape3D> small_full_B_ptr = get_full_B_ptr()->clone();
  // make it smaller 
  small_full_B_ptr ->scale_around_origin(CartesianCoordinate3D<float>(1.F,fraction,fraction));

  // the same for D
  shared_ptr<Shape3D> large_D_ptr = get_D_ptr()->clone();
  large_D_ptr ->scale_around_origin(CartesianCoordinate3D<float>(1.F,1/fraction,1/fraction));
  // and E
  shared_ptr<Shape3D> large_E_ptr = get_E_ptr()->clone();
  large_E_ptr ->scale_around_origin(CartesianCoordinate3D<float>(1.F,1/fraction,1/fraction));
  
  // combine the whole thing
  shared_ptr<Shape3D> small_full_B_notD_ptr = 
    new CombinedShape3D< logical_and_not<bool> > ( small_full_B_ptr,large_D_ptr);
  return 
    new CombinedShape3D< logical_and_not<bool> > ( small_full_B_notD_ptr,large_E_ptr);
}


shared_ptr<Shape3D> 
Utah_phantom::make_inside_C_ptr(const float fraction) const
{
  shared_ptr<Shape3D> small_full_C_ptr = get_full_C_ptr()->clone();
  small_full_C_ptr ->scale_around_origin(CartesianCoordinate3D<float>(1.F,fraction,fraction));
  shared_ptr<Shape3D> large_full_B_ptr = get_full_B_ptr()->clone();
  large_full_B_ptr ->scale_around_origin(CartesianCoordinate3D<float>(1.F,1/fraction,1/fraction));
  return 
    new CombinedShape3D< logical_and_not<bool> > ( small_full_C_ptr,large_full_B_ptr);
}
void 
Utah_phantom::translate(const CartesianCoordinate3D<float>& direction)
{
  A_ptr->translate(direction);
  B_ptr->translate(direction);
  full_B_ptr->translate(direction);
  C_ptr->translate(direction);
  full_C_ptr->translate(direction);
  D_ptr->translate(direction);
  E_ptr->translate(direction);
}

void 
Utah_phantom::scale(const CartesianCoordinate3D<float>& scale3D)
{
  A_ptr->scale(scale3D);
  B_ptr->scale(scale3D);
  full_B_ptr->scale(scale3D);
  C_ptr->scale(scale3D);
  full_C_ptr->scale(scale3D);
  D_ptr->scale(scale3D);
  E_ptr->scale(scale3D);
}

END_NAMESPACE_STIR

#endif
