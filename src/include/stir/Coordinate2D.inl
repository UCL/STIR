//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license
    See STIR/LICENSE.txt for details
*/
/*!
  \file

  \brief inline implementations for the stir::Coordinate2D class

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project


*/

START_NAMESPACE_STIR

template <class coordT>
Coordinate2D<coordT>::Coordinate2D()
    : base_type()
{}

template <class coordT>
Coordinate2D<coordT>::Coordinate2D(const coordT& c1, const coordT& c2)
    : base_type()
{
  (*this)[1] = c1;
  (*this)[2] = c2;
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
