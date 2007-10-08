//
//$Id$
//
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
/*!
  \file 
  \ingroup Coordinate  
  \brief inline implementations for the stir::Coordinate4D class 

  \author Sanida Mustafovic 
  \author Kris Thielemans 
  \author PARAPET project

  $Date$
  $Revision$

*/

START_NAMESPACE_STIR

template <class coordT>
Coordinate4D<coordT>::Coordinate4D()
  : base_type()
{}

template <class coordT>
Coordinate4D<coordT>::Coordinate4D(const coordT& c1, 
				   const coordT& c2, 
				   const coordT& c3, 
				   const coordT& c4)
  : base_type()
{
  (*this)[1] = c1;
  (*this)[2] = c2;
  (*this)[3] = c3;
  (*this)[4] = c4;
}

template <class coordT>
Coordinate4D<coordT>::Coordinate4D(const base_type& c)
  : base_type(c)
{}

template <class coordT>
Coordinate4D<coordT>&
Coordinate4D<coordT>::operator=(const base_type& c)
{
  base_type::operator=(c);
  return *this;
}


END_NAMESPACE_STIR
