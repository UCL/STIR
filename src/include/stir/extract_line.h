//
//
/*
    Copyright (C) 2004- 2009, Hammersmith Imanet
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_extract_line_H__
#define __stir_extract_line_H__
/*!
  \file
  \ingroup buildblock
  \brief Declaration  of stir::extract_line

  \author Charalampos Tsoumpas
  \author Kris Thielemans


 */
#include "stir/Array.h"
#include "stir/BasicCoordinate.h"
START_NAMESPACE_STIR

/*!
   \ingroup buildblock
   \brief  extracts a line from an array in the direction of the specified dimension.
   \todo make n-dimensional version
*/
template <class elemT>
Array<1, elemT> inline extract_line(const Array<3, elemT>&, const BasicCoordinate<3, int>& index, const int dimension);
END_NAMESPACE_STIR

#include "stir/extract_line.inl"
#endif
