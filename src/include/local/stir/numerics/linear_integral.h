//
// $Id$
//
/*
    Copyright (C) 2004 - $Date$, Hammersmith Imanet Ltd
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
  \brief Declaration of linear_integral functions 

  \author Charalampos Tsoumpas
 
  $Date$
  $Revision$
*/

#ifndef __stir_linear_integral_H__
#define __stir_linear_integral_H__

#include "stir/common.h"
#include <vector>
#include <iostream> 
#include <cstring>
#include <iomanip> 
#include <fstream>

START_NAMESPACE_STIR

inline float 
linear_integral(std::vector<float> f , std::vector<float> t, int approx);

END_NAMESPACE_STIR

#include "local/stir/numerics/linear_integral.inl"

#endif //__linear_integral_H__
