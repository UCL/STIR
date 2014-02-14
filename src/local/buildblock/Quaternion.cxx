//
//
/*!
  \file
  \ingroup buildblock

  \brief Implementation of class Quaternion

  \author: Sanida Mustafovic
  \author: Kris Thielemans
*/

/*
    Copyright (C) 2000- 2004, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/


#include "local/stir/Quaternion.h"


START_NAMESPACE_STIR

template <class coordT>
Quaternion<coordT>::Quaternion()
  : base_type()
{}

template <class coordT>
Quaternion<coordT>::Quaternion(const coordT& c1, 
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
Quaternion<coordT>::Quaternion(const base_type& c)
  : base_type(c)
{}

#if 0
template <class coordT>
coordT
Quaternion<coordT>:: component_1() const
{
  return (*this)[1];
}

template <class coordT>
coordT
Quaternion<coordT>:: component_2() const
{
  return (*this)[2];
}

template <class coordT>
coordT
Quaternion<coordT>:: component_3() const
{
  return (*this)[3];
}

template <class coordT>
coordT
Quaternion<coordT>:: component_4() const
{
  return (*this)[4];
}

#endif
   

template class Quaternion<float>;
END_NAMESPACE_STIR
