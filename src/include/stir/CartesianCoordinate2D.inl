//
// $Id$
//

/*!
  \file 
  \ingroup Coordinate
 
  \brief inline implementations for the stir::CartesianCoordinate2D<coordT> class 

  \author Kris Thielemans 
  \author PARAPET project

  $Date$

  $Revision$

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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
CartesianCoordinate2D<coordT>::CartesianCoordinate2D()
  : Coordinate2D<coordT>()
{}

template <class coordT>
CartesianCoordinate2D<coordT>::CartesianCoordinate2D(const coordT& y, 
						     const coordT& x)
  : Coordinate2D<coordT>(y,x)
{}


template <class coordT>
CartesianCoordinate2D<coordT>::CartesianCoordinate2D(const BasicCoordinate<2,coordT>& c)
  : base_type(c)
{}



template <class coordT>
CartesianCoordinate2D<coordT>& 
CartesianCoordinate2D<coordT>:: operator=(const BasicCoordinate<2,coordT>& c)
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
