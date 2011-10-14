//
// $Id$
//

/*!
  \file 
  \ingroup Coordinate  
  \brief inline implementations for the stir::CartesianCoordinate3D<coordT> class 

  \author Sanida Mustafovic 
  \author Kris Thielemans 
  \author PARAPET project

  $Date$

  $Revision$

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009-06-22, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - $Date$, Kris Thielemans
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


START_NAMESPACE_STIR

template <class coordT>
CartesianCoordinate3D<coordT>::CartesianCoordinate3D()
  : Coordinate3D<coordT>()
{}

template <class coordT>
CartesianCoordinate3D<coordT>::CartesianCoordinate3D(const coordT& z, 
						     const coordT& y, 
						     const coordT& x)
  : Coordinate3D<coordT>(z,y,x)
{}


template <class coordT>
CartesianCoordinate3D<coordT>::CartesianCoordinate3D(const BasicCoordinate<3, coordT>& c)
  : base_type(c)
{}



template <class coordT>
CartesianCoordinate3D<coordT>& 
CartesianCoordinate3D<coordT>:: operator=(const BasicCoordinate<3, coordT>& c)
{
  basebase_type::operator=(c);
  return *this;
}

#ifdef OLDDESIGN
template <class coordT>
CartesianCoordinate3D<coordT> ::CartesianCoordinate3D(const Point3D& p)

{
  x() = p.x;
  y() = p.y;
  z() = p.z;
}
#endif

template <class coordT>
coordT&
CartesianCoordinate3D<coordT>::z()
{
  return this->operator[](1);
}


template <class coordT>
coordT
CartesianCoordinate3D<coordT>::z() const
{
  return this->operator[](1);
}


template <class coordT>
coordT&
CartesianCoordinate3D<coordT>::y()
{
  return this->operator[](2);
}


template <class coordT>
coordT
CartesianCoordinate3D<coordT>::y() const
{
  return this->operator[](2);
}


template <class coordT>
coordT&
CartesianCoordinate3D<coordT>::x()
{
  return this->operator[](3);
}


template <class coordT>
coordT
CartesianCoordinate3D<coordT>::x() const
{
  return this->operator[](3);
}


END_NAMESPACE_STIR
