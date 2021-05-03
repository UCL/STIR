//
//
/*!

  \file
  \ingroup Array
  \brief Declaration of class stir::BoundaryConditions

  \author Kris Thielemans

*/
/*
    Copyright (C) 2009- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/common.h"

START_NAMESPACE_STIR

/*! \ingroup Array
  \brief Preliminary class to specify boundary conditions for filters
*/
class BoundaryConditions{
 public:
  enum BC {zero, constant, periodic};
};

END_NAMESPACE_STIR
