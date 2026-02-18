//
//
/*
    Copyright (C) 2004- 2007, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_numerics_determinant_H__
#define __stir_numerics_determinant_H__
/*!
  \file
  \ingroup numerics

  \brief Declaration of stir::determinant() function for matrices

  \author Kris Thielemans

*/
#include "stir/ArrayFwd.h"

START_NAMESPACE_STIR

/*! \ingroup numerics
  \brief Compute the determinant of a matrix

  Matrix indices can start from any number.

  \todo Only works for low dimensions for now.
*/
template <class elemT>
elemT determinant(const Array<2, elemT>& m);

END_NAMESPACE_STIR
#endif
