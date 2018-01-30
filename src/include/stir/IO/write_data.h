#ifndef __stir_IO_write_data_H__
#define __stir_IO_write_data_H__
/*
    Copyright (C) 2004- 2007, Hammersmith Imanet Ltd
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
/*!
  \file 
  \ingroup Array_IO 
  \brief declarations of stir::write_data() functions for writing Arrays to file

  \author Kris Thielemans

*/

#include "stir/ByteOrder.h"
START_NAMESPACE_STIR

class Succeeded;
class NumericType;
template <class T> class NumericInfo;
template <int num_dimensions, class elemT> class Array;

#if defined(_MSC_VER) && _MSC_VER==1200
// VC 6.0 cannot compile this when the templates are declared first, 
// and defined in the .inl
#  define __STIR_WORKAROUND_TEMPLATES 1
#endif


/*
   If your still compiler cannot handle this code, you can try to put

#  define __STIR_WORKAROUND_TEMPLATES 2

   This enables a horrible work-around where the functions are 
   not templated in terms of num_dimensions. Instead, some 
   preprocessor trickery is used.
   See the end of this file and write_data.inl for more info.
*/

#ifndef __STIR_WORKAROUND_TEMPLATES
/*! \ingroup Array_IO
  \brief Write the data of an Array to file.

  Only the data will be written, not the dimensions, start indices, nor byte-order.
  Hence, this should only used for low-level IO.

  \a OStreamT is supposed to be stream or file type (see implementations for 
  detail::write_data_1d()).

  If \a can_corrupt_data = \c true, the data in the array can be 
  byte-swapped after writing, and hence should no longer be used.
  This is useful for saving (a little bit of) time.

  \warning When an error occurs, the function immediately returns. 
  However, the data might have been partially written to \a s.

  \warning For deficient compilers (e.g. VC 6.0), work-arounds are
  necessary to get this to compile. In particular, this function
  is then not templated in \c num_dimensions, but explicitly
  defined for a few dimensions (see the source).
*/
template <int num_dimensions,  class OStreamT, class elemT>
inline Succeeded 
write_data(OStreamT& s, const Array<num_dimensions,elemT>& data, 
	   const ByteOrder byte_order=ByteOrder::native,
	   const bool can_corrupt_data=false);
/*! \ingroup Array_IO
  \brief Write the data of an Array to file as a different type.

  This function essentially first calls convert_data() to construct
  an array with elements of type \a OutputType, and then calls 
  write_data(OstreamT&, const Array<num_dimensions,elemT>&, 
	   const ByteOrder, const bool).
  \see write_data(OstreamT&, const Array<num_dimensions,elemT>&, 
	   const ByteOrder, const bool)

  \see find_scale_factor() for the meaning of \a scale_factor.
  \warning For deficient compilers (e.g. VC 6.0), work-arounds are
  necessary to get this to compile. In particular, this function
  is then not templated in \c num_dimensions, but explicitly
  defined for a few dimensions (see the source).
*/
template <int num_dimensions,  class OStreamT, class elemT, class OutputType, class ScaleT>
inline Succeeded 
write_data(OStreamT& s, const Array<num_dimensions,elemT>& data, 
	   NumericInfo<OutputType> output_type, 
	   ScaleT& scale_factor,
	   const ByteOrder byte_order=ByteOrder::native,
	   const bool can_corrupt_data=false);

/*! \ingroup Array_IO
  \brief Write the data of an Array to file as a different type but using a given scale factor.
  
  If \a scale_factor is such that the <tt>data/scale_factor</tt> does not fit in the
  range for \a OutputType, the writing will fail. However, data might have been
  partially written to file anyway.

  \see write_data(OStreamT&, const Array<num_dimensions,elemT>&, 
	   NumericInfo<OutputType>, 
	   ScaleT&,
	   const ByteOrder,
	   const bool)
  \warning For deficient compilers (e.g. VC 6.0), work-arounds are
  necessary to get this to compile. In particular, this function
  is then not templated in \c num_dimensions, but explicitly
  defined for a few dimensions (see the source).
*/
template <int num_dimensions,  class OStreamT, class elemT, class OutputType, class ScaleT>
inline Succeeded 
write_data_with_fixed_scale_factor(OStreamT& s, const Array<num_dimensions,elemT>& data, 
				   NumericInfo<OutputType> output_type, 
				   const ScaleT scale_factor,
				   const ByteOrder byte_order=ByteOrder::native,
				   const bool can_corrupt_data=false);

/*! \ingroup Array_IO
  \brief Write the data of an Array to file as a different type.

  \see write_data(OStreamT&, const Array<num_dimensions,elemT>&, 
	   NumericInfo<OutputType>, 
	   ScaleT&,
	   const ByteOrder,
	   const bool)
  The only difference is that the output type is now specified using NumericType.

  \warning For deficient compilers (e.g. VC 6.0), work-arounds are
  necessary to get this to compile. In particular, this function
  is then not templated in \c num_dimensions, but explicitly
  defined for a few dimensions (see the source).
*/

template <int num_dimensions,  class OStreamT, class elemT, class ScaleT>
inline Succeeded 
write_data(OStreamT& s, 
	   const Array<num_dimensions,elemT>& data, 
	   NumericType type, ScaleT& scale,
	   const ByteOrder byte_order=ByteOrder::native,
	   const bool can_corrupt_data=false);

#endif //__STIR_WORKAROUND_TEMPLATES



END_NAMESPACE_STIR


#if !defined(__STIR_WORKAROUND_TEMPLATES) || __STIR_WORKAROUND_TEMPLATES<2

#include "stir/IO/write_data.inl"

#else

#define num_dimensions 1
#include "stir/IO/write_data.inl"
#undef num_dimensions
#define num_dimensions 2
#include "stir/IO/write_data.inl"
#undef num_dimensions
#define num_dimensions 3
#include "stir/IO/write_data.inl"
#undef num_dimensions
//#define num_dimensions 4
//#include "stir/IO/write_data.inl"
//#undef num_dimensions

#endif

#ifdef __STIR_WORKAROUND_TEMPLATES
#  undef __STIR_WORKAROUND_TEMPLATES
#endif

#endif
