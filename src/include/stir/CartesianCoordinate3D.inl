//
// $Id$
//

/*!
  \file 
  \ingroup Coordinate  
  \brief inline implementations for the CartesianCoordinate3D<coordT> class 

  \author Sanida Mustafovic 
  \author Kris Thielemans 
  \author PARAPET project

  $Date$

  $Revision$

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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
CartesianCoordinate3D<coordT>::CartesianCoordinate3D(const basebase_type& c)
  : base_type(c)
{}



template <class coordT>
CartesianCoordinate3D<coordT>& 
CartesianCoordinate3D<coordT>:: operator=(const basebase_type& c)
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
