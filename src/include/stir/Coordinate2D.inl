//
// $Id$
//

/*!
  \file 
 
  \brief inline implementations for the Coordinate2D<coordT> class 

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
Coordinate2D<coordT>::Coordinate2D()
  : base_type()
{}

template <class coordT>
Coordinate2D<coordT>::Coordinate2D(const coordT& c1, 
				   const coordT& c2)
  : base_type()
{
  coords[1] = c1;
  coords[2] = c2;
}

template <class coordT>
Coordinate2D<coordT>::Coordinate2D(const base_type& c)
  : base_type(c)
{}

template <class coordT>
Coordinate2D<coordT>&
Coordinate2D<coordT>::operator=(const base_type& c)
{
  base_type::operator=(c);
  return *this;
}


END_NAMESPACE_STIR
