// $Id$
#ifndef __stir_write_data_H__
#define __stir_write_data_H__

/*!
  \file 
  \ingroup Array 
  \brief declarations of write_data() functions for writing Arrays to file

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/

#include "stir/ByteOrder.h"
#include <stdio.h>
#include <iostream>
START_NAMESPACE_STIR

class Succeeded;
class NumericType;
template <class T> class NumericInfo;
template <int num_dimensions, class elemT> class Array;

/*! \ingroup Array
  \brief Write the data of an Array to file.

  Only the data will be written, not the dimensions, start indices, nor byte-order.
  Hence, this should only used for low-level IO.

  \a OStreamT is supposed to be stream or file type (see implementations for 
  write_data_1d()).

  If \a can_corrupt_data = \c true, the data in the array can be 
  byte-swapped after writing, and hence should no longer be used.
  This is useful for saving (a little bit of) time.

  \warning When an error occurs, the function immediately returns. 
  However, the data might have been partially written to \a s.
*/
template <class OStreamT, int num_dimensions, class elemT>
inline Succeeded 
write_data(OStreamT& s, const Array<num_dimensions,elemT>& data, 
	   const ByteOrder byte_order=ByteOrder::native,
	   const bool can_corrupt_data=false);

/*! \ingroup Array
  \brief Write the data of an Array to file as a different type.

  This function essentially first calls convert_data() to construct
  an array with elements of type \a OutputType, and then calls 
  write_data(OstreamT&, const Array<num_dimensions,elemT>&, 
	   const ByteOrder, const bool).
  \see write_data(OstreamT&, const Array<num_dimensions,elemT>&, 
	   const ByteOrder, const bool)

  \see find_scale_factor() for the meaning of \a scale_factor.
*/
template <class OStreamT, int num_dimensions, class elemT, class OutputType, class ScaleT>
inline Succeeded 
write_data(OStreamT& s, const Array<num_dimensions,elemT>& data, 
	   NumericInfo<OutputType> output_type, 
	   ScaleT& scale_factor,
	   const ByteOrder byte_order=ByteOrder::native,
	   const bool can_corrupt_data=false);

/*! \ingroup Array
  \brief Write the data of an Array to file as a different type but using a given scale factor.
  
  If \a scale_factor is such that the <tt>data/scale_factor</tt> does not fit in the
  range for \a OutputType, the writing will fail. However, data might have been
  partially written to file anyway.

  \see write_data(OStreamT&, const Array<num_dimensions,elemT>&, 
	   NumericInfo<OutputType>, 
	   ScaleT&,
	   const ByteOrder,
	   const bool)
*/
template <class OStreamT, int num_dimensions, class elemT, class OutputType, class ScaleT>
inline Succeeded 
write_data_with_fixed_scale_factor(OStreamT& s, const Array<num_dimensions,elemT>& data, 
				   NumericInfo<OutputType> output_type, 
				   const ScaleT scale_factor,
				   const ByteOrder byte_order=ByteOrder::native,
				   const bool can_corrupt_data=false);

/*! \ingroup Array
  \brief Write the data of an Array to file as a different type.

  \see write_data(OStreamT&, const Array<num_dimensions,elemT>&, 
	   NumericInfo<OutputType>, 
	   ScaleT&,
	   const ByteOrder,
	   const bool)
  The only difference is that the output type is now specified using NumericType.
*/
template <class OStreamT, int num_dimensions, class elemT, class ScaleT>
inline Succeeded 
write_data(OStreamT& s, 
	   const Array<num_dimensions,elemT>& data, 
	   NumericType type, ScaleT& scale,
	   const ByteOrder byte_order=ByteOrder::native,
	   const bool can_corrupt_data=false);

END_NAMESPACE_STIR

#include "stir/write_data.inl"

#endif
