// $Id$
#ifndef __stir_IO_read_data_1d_H__
#define __stir_IO_read_data_1d_H__
/*!
  \file 
  \ingroup Array_IO_detail 
  \brief Declaration of read_data_1d() functions for reading Arrays to file

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
