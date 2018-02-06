//
//
/*
    Copyright (C) 2006 - 2009, Hammersmith Imanet Ltd
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
/*!
  \file
  \ingroup modelling

  \brief Declaration of class stir::KineticParameters

  \author Charalampos Tsoumpas
 
*/

#ifndef __stir_modelling_KineticParameters_H__
#define __stir_modelling_KineticParameters_H__

#include "stir/BasicCoordinate.h"
#include "stir/assign.h"

START_NAMESPACE_STIR

template <int num_param, typename elemT>
class KineticParameters:public BasicCoordinate<num_param,elemT>
{
  typedef BasicCoordinate<num_param,elemT>  base_type;
 public:
  KineticParameters()
    {}

  KineticParameters(const base_type& c)
    : base_type(c)
    {}

};

template <int num_dimensions, class T, class T2>
inline 
void assign(KineticParameters<num_dimensions,T>& v, const T2& y)
{
  for (typename KineticParameters<num_dimensions,T>::iterator iter = v.begin(); 
       iter != v.end(); ++iter)
    assign(*iter, y);
}

END_NAMESPACE_STIR

//#include "stir/modelling/KineticParameters.inl"

#endif //__stir_modelling_KineticParameters_H__
