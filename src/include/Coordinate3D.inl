//
// $Id$: $Date$
//

/*!
  \file 
 
  \brief inline implementations for the Coordinate3D<coordT> class 

  \author Sanida Mustafovic 
  \author Kris Thielemans 
  \author PARAPET project

  \date    $Date$

  \version $Revision$

*/



START_NAMESPACE_TOMO

template <class coordT>
Coordinate3D<coordT>::Coordinate3D()
  : base_type()
{}

template <class coordT>
Coordinate3D<coordT>::Coordinate3D(const coordT& c1, 
				   const coordT& c2, 
				   const coordT& c3)
  : base_type()
{
  coords[1] = c1;
  coords[2] = c2;
  coords[3] = c3;
}

template <class coordT>
Coordinate3D<coordT>::Coordinate3D(const base_type& c)
  : base_type(c)
{}

template <class coordT>
Coordinate3D<coordT>&
Coordinate3D<coordT>::operator=(const base_type& c)
{
  base_type::operator=(c);
  return *this;
}


END_NAMESPACE_TOMO
