//
// $Id$
//
/*!

  \file
  \ingroup Array
  \brief Declaration of class stir::BoundaryConditions

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2009- $Date$, Hammersmith Imanet Ltd
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
