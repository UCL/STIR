// $Id$
#ifndef __stir_IO_read_data_1d_H__
#define __stir_IO_read_data_1d_H__
/*!
  \file 
  \ingroup Array_IO_detail 
  \brief Declaration of stir::read_data_1d() functions for reading 1D stir::Array objects from file

  \author Kris Thielemans

  $Date$
  $Revision$
*/
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
#include "stir/common.h"
#include <stdio.h>
#include <iostream>

START_NAMESPACE_STIR
class Succeeded;
class ByteOrder;
template <int num_dimensions, class elemT> class Array;

namespace detail {
/*! \ingroup Array_IO_detail
  \brief This is an internal function called by \c read_data(). It does the actual reading
   to \c std::istream.

  This function might propagate any exceptions by std::istream::read.
 */
template <class elemT>
inline Succeeded
read_data_1d(std::istream& s, Array<1, elemT>& data,
	     const ByteOrder byte_order);


/* \ingroup Array_IO_detail
  \brief  This is the (internal) function that does the actual reading from a FILE*.
  \internal
 */
template <class elemT>
inline Succeeded
read_data_1d(FILE*& , Array<1, elemT>& data,
	     const ByteOrder byte_order);

} // end namespace detail
END_NAMESPACE_STIR

#include "stir/IO/read_data_1d.inl"

#endif
