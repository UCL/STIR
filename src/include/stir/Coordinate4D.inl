//
//$Id$
//


/*!
  \file 
  \ingroup Coordinate  
  \brief inline implementations for the Coordinate4D<coordT> class 

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
  this->coords[1] = c1;
  this->coords[2] = c2;
  this->coords[3] = c3;
  this->coords[4] = c4;
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
