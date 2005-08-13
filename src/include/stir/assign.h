
// $Id$
/*
    Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
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

#ifndef __stir_assign_H__
#define __stir_assign_H__

/*!
  \file 
  \ingroup buildblock
  \brief defines the stir::assign function to assign values to different data types

  \author Kris Thielemans 

  $Date$
  $Revision$
*/
#include "stir/VectorWithOffset.h"
#include "stir/ByteOrder.h"
#include "stir/BasicCoordinate.h"   
#include <vector>

START_NAMESPACE_STIR

// TODO hopefully next ifdef is not necessary. Otherwise we need to have more for ints etc
#if defined(_MSC_VER) && _MSC_VER<=1300
inline 
void assign(double& x, const double y)
{
  x=y;
}

static inline 
void assign(float& x, const float y)
{
  x=y;
}
#else

template <class T, class T2>
  inline 
  void assign(T& x, const T2& y)
{
  x=y;
}
#endif

template <class T, class T2>
inline 
void assign(std::vector<T>& v, const T2& y)
{
  for (typename std::vector<T>::iterator iter = v.begin(); 
       iter != v.end(); ++iter)
    assign(*iter, y);
}

template <int num_dimensions, class T, class T2>
inline 
void assign(BasicCoordinate<num_dimensions,T>& v, const T2& y)
{
  for (typename BasicCoordinate<num_dimensions,T>::iterator iter = v.begin(); 
       iter != v.end(); ++iter)
    assign(*iter, y);
}

template <class T, class T2>
inline 
void assign(VectorWithOffset<T>& v, const T2& y)
{
  for (typename VectorWithOffset<T>::iterator iter = v.begin(); 
       iter != v.end(); ++iter)
    assign(*iter, y);
}

END_NAMESPACE_STIR

#endif
