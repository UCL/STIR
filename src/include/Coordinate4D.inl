//
// $Id$: $Date$
//


/*
  inline implementations for Coordinate4D methods
  
  History:
   1.0 (25/01/2000)
     Kris Thielemans and Alexey Zverovich
*/


START_NAMESPACE_TOMO

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
  coords[1] = c1;
  coords[2] = c2;
  coords[3] = c3;
  coords[4] = c4;
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


END_NAMESPACE_TOMO
