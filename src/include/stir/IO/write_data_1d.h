// $Id$
#ifndef __stir_IO_write_data_1d_H__
#define __stir_IO_write_data_1d_H__
/*!
  \file 
  \ingroup Array_IO_detail 
  \brief Declaration of write_data_1d() functions for writing Arrays to file

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
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
