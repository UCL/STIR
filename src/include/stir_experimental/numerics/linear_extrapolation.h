//
//
/*
    Copyright (C) 2005- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file 
  \ingroup numerics
  \brief stir::linear_extrapolation

  \author Charalampos Tsoumpas

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
