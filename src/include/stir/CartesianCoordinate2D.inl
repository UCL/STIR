//
// $Id$: $Date$
//

/*!
  \file 
  \ingroup buildblock
 
  \brief inline implementations for the CartesianCoordinate2D<coordT> class 

  \author Kris Thielemans 
  \author PARAPET project

  \date    $Date$

  \version $Revision$

*/

START_NAMESPACE_TOMO

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
  return operator[](1);
}


template <class coordT>
coordT
CartesianCoordinate2D<coordT>::y() const
{
  return operator[](1);
}


template <class coordT>
coordT&
CartesianCoordinate2D<coordT>::x()
{
  return operator[](2);
}


template <class coordT>
coordT
CartesianCoordinate2D<coordT>::x() const
{
  return operator[](2);
}


END_NAMESPACE_TOMO
