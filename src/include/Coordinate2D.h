//
// $Id$: $Date$
//
#ifndef __Coordinate2D_H__
#define __Coordinate2D_H__
/*!
  \file 
  \ingroup buildblock 
  \brief defines the Coordinate2D<coordT> class 

  \author Sanida Mustafovic 
  \author Kris Thielemans 
  \author PARAPET project

  \date    $Date$

  \version $Revision$

*/

/*!
  \ingroup buildblock 
  \brief a templated class for 2-dimensional coordinates.

   The only new method is a constructor Coordinate2D<coordT>(c1,c2)

   \warning Indices run from 1 to 2. 

*/

#include "BasicCoordinate.h"

START_NAMESPACE_TOMO

template <typename coordT>
class Coordinate2D : public BasicCoordinate<2, coordT>
{
protected:
  typedef BasicCoordinate<2, coordT> base_type;

public:
  inline Coordinate2D();
  inline Coordinate2D(const coordT&, const coordT&);
  inline Coordinate2D(const base_type& c);
  inline Coordinate2D& operator=(const base_type& c);
};

END_NAMESPACE_TOMO

#include "Coordinate2D.inl"

#endif

