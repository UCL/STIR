#ifndef __stir_IO_read_data_1d_H__
#define __stir_IO_read_data_1d_H__
/*!
  \file
  \ingroup Array_IO_detail
  \brief Declaration of stir::read_data_1d() functions for reading 1D stir::Array objects from file

  \author Kris Thielemans

*/
/*
    Copyright (C) 2004- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#include "stir/ArrayFwd.h"
#include <stdio.h>
#include <iostream>

START_NAMESPACE_STIR
class Succeeded;
class ByteOrder;

namespace detail
{
/*! \ingroup Array_IO_detail
  \brief This is an internal function called by \c read_data(). It does the actual reading
   to \c std::istream.

  This function might propagate any exceptions by std::istream::read.
 */
template <int num_dimensions, class elemT>
inline Succeeded read_data_1d(std::istream& s, ArrayType<num_dimensions, elemT>& data, const ByteOrder byte_order);

/* \ingroup Array_IO_detail
  \brief  This is the (internal) function that does the actual reading from a FILE*.
  \internal
 */
template <int num_dimensions, class elemT>
inline Succeeded read_data_1d(FILE*&, ArrayType<num_dimensions, elemT>& data, const ByteOrder byte_order);

} // end namespace detail
END_NAMESPACE_STIR

#include "stir/IO/read_data_1d.inl"

#endif
