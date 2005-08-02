//
// $Id$
//
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
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
#ifndef __stir_numerics_MatrixFunction_H__
#define __stir_numerics_MatrixFunction_H__
/*!
  \file
  \ingroup numerics
  
  \brief Declaration of stir::determinant() function for matrices
    
  \author Kris Thielemans

  $Date$
  $Revision$
*/
#include "stir/common.h"

START_NAMESPACE_STIR

template <int num_dimensions, class elemT> class Array;

/*! \ingroup numerics
  \brief Compute the determinant of a matrix

  \todo Only works for low dimensions for now.
*/
template <class elemT>
elemT
determinant(const Array<2,elemT>& m);

END_NAMESPACE_STIR
#endif
