//
//
/*
    Copyright (C) 2006 - 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

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
