#ifndef __stir_IO_write_data_1d_H__
#define __stir_IO_write_data_1d_H__
/*!
  \file 
  \ingroup Array_IO_detail 
  \brief Declaration of stir::write_data_1d() functions for writing 1D stir::Array objects to file

  \author Kris Thielemans

*/
/*
    Copyright (C) 2004- 2009, Hammersmith Imanet Ltd
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
  \brief This is an internal function called by \c write_data(). It does the actual writing
   to \c std::ostream.

  This function does not throw any exceptions. Exceptions thrown by std::ostream::write 
  are caught.
 */
template <class elemT>
inline Succeeded
write_data_1d(std::ostream& s, const Array<1, elemT>& data,
	      const ByteOrder byte_order,
	      const bool can_corrupt_data);
/*! \ingroup Array_IO_detail
  \brief This is an internal function called by \c write_data(). It does the actual writing
   to \c FILE* using stdio functions.

  This function does not throw any exceptions. 
  
 */
template <class elemT>
inline Succeeded
write_data_1d(FILE* & fptr_ref, const Array<1, elemT>& data,
	   const ByteOrder byte_order,
	   const bool can_corrupt_data);
}

END_NAMESPACE_STIR

#include "stir/IO/write_data_1d.inl"

#endif
