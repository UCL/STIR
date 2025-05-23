//
//
/*
    Copyright (C) 2004 - 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

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
