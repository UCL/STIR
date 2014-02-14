//
//
/*
    Copyright (C) 2004 - 2009, Hammersmith Imanet Ltd
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
  \ingroup numerics
  \brief Declaration of stir::integrate_discrete_function function

  \author Charalampos Tsoumpas
 
*/

#ifndef __stir_integrate_discrete_function_H__
#define __stir_integrate_discrete_function_H__

#include "stir/common.h"
#include <vector>
#include <iostream> 
#include <cstring>
#include <iomanip> 
#include <fstream>

START_NAMESPACE_STIR
/*!
  \brief numerical integration of a 1D function
  \ingroup numerics

  This is a simple integral implementation using rectangular (=0) or trapezoidal (=1) approximation.
  It currently integrates over the complete range specified.

  \param coordinates Coordinates at which the function samples are given
  \param values Function values
  \param interpolation_order has to be 0 or 1
  \warning Type \c elemT should not be an integral type.
*/
template <typename elemT>
inline elemT 
integrate_discrete_function(const std::vector<elemT> & coordinates, const std::vector<elemT> & values, const int interpolation_order = 1);

END_NAMESPACE_STIR

#include "stir/numerics/integrate_discrete_function.inl"

#endif //__integrate_discrete_function_H__
