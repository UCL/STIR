//
// $Id$: $Date$
//


/*
  inline implementations for Coordinate2D methods
  
  History:
   1.0 (25/01/2000)
     Kris Thielemans and Alexey Zverovich
*/


START_NAMESPACE_TOMO

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


END_NAMESPACE_TOMO
