//
// $Id$
//

/*!
  \file 
  \ingroup Coordinate
 
  \brief inline implementations for the CartesianCoordinate2D<coordT> class 

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
CartesianCoordinate2D<coordT>::CartesianCoordinate2D()
  : Coordinate2D<coordT>()
{}

template <class coordT>
CartesianCoordinate2D<coordT>::CartesianCoordinate2D(const coordT& y, 
						     const coordT& x)
  : Coordinate2D<coordT>(y,x)
{}


template <class coordT>
CartesianCoordinate2D<coordT>::CartesianCoordinate2D(const basebase_type& c)
  : base_type(c)
{}



template <class coordT>
CartesianCoordinate2D<coordT>& 
CartesianCoordinate2D<coordT>:: operator=(const basebase_type& c)
{
  basebase_type::operator=(c);
  return *this;
}


template <class coordT>
coordT&
CartesianCoordinate2D<coordT>::y()
{
  return this->operator[](1);
}


template <class coordT>
coordT
CartesianCoordinate2D<coordT>::y() const
{
  return this->operator[](1);
}


template <class coordT>
coordT&
CartesianCoordinate2D<coordT>::x()
{
  return this->operator[](2);
}


template <class coordT>
coordT
CartesianCoordinate2D<coordT>::x() const
{
  return this->operator[](2);
}


END_NAMESPACE_STIR
