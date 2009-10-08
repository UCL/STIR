//
// $Id$
//
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
/*!
  \file 
  \ingroup numerics
  \brief stir::linear_extrapolation

  \author Charalampos Tsoumpas

  $Date$
  $Revision$
*/


#include "stir/common.h"
START_NAMESPACE_STIR


  template <typename in_elemT>
  inline void
  linear_extrapolation(std::vector<in_elemT> &input_vector) 
  {
    input_vector.push_back(*(input_vector.end()-1)*2 - *(input_vector.end()-2));
    input_vector.insert(input_vector.begin(), *input_vector.begin()*2 - *(input_vector.begin()+1));
  }

END_NAMESPACE_STIR
