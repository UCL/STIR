//
// $Id$
//
#ifndef __convert_array_H__
#define  __convert_array_H__

/*!
  \file 
  \ingroup buildblock
 
  \brief This file declares the convert_array templates.
  This are functions to convert Array objects to a different numeric type.

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/

// TODO enable FULL 

#include "stir/NumericInfo.h"
#include "stir/Array.h"

START_NAMESPACE_STIR

/*!
  \ingroup buildblock
   \brief A function that finds a scale factor to use when converting data to a new type

   The scale factor is such that 
   (\a data_in / \a scale_factor)
   will fit in the maximum range for the output type.

   When input and output types are identical, \a scale_factor is set to 1.

   \param scale_factor 
          a reference to a (float or double) variable which will be
	  set to the scale factor such that (ignoring types)
	   \code  data_in == data_out * scale_factor \endcode
	  If scale_factor is initialised to 0, the maximum range of \a T2
	  is used. If scale_factor != 0, find_scale_factor attempts to use the
	  given scale_factor, unless the T2 range doesn't fit.
	  In that case, the same scale_factor is used as in the 0 case.
   
   \param data_in 
          some Array object, elements are of some numeric type \a T1
   \param info2
          \a T2 is the desired output type

   Note that there is an effective threshold at 0 currently (i.e. negative
   numbers are ignored) when \a T2 is an unsigned type.

   \see convert_array
*/
template <int num_dimensions, class T1, class T2, class scaleT>
void
find_scale_factor(scaleT& scale_factor,
		  const Array<num_dimensions,T1>& data_in, 
		  const NumericInfo<T2> info_for_out_type);
/*!
  \ingroup buildblock
   \brief A function that returns a new Array (of the same dimension) with elements of type \c T2

   Result is (approximately) \a data_in / \a scale_factor.

   \par example 
   \code
   Array<2,float> data_out = convert_array(scale_factor, data_in, NumericInfo<float>())
   \endcode

   \param scale_factor 
          a reference to a (float or double) variable which will be
	  set to the scale factor such that (ignoring types)
	   \code  data_in == data_out * scale_factor \endcode
   
   \param data_in 
          some Array object, elements are of some numeric type \a T1
   \param info2
          \a T2 is the desired output type

   \return 
      data_out :
          an Array object whose elements are of numeric type T2.

   When the output type is integer, rounding is used.

   Note that there is an effective threshold at 0 currently (i.e. negative
   numbers are cut out) when T2 is an unsigned type.

  \see find_scale_factor for more info on the determination of \a scale_factor.
*/
#if 1
template <int num_dimensions, class T1, class T2, class scaleT>
Array<num_dimensions, T2>
convert_array(scaleT& scale_factor,
	      const Array<num_dimensions, T1>& data_in, 
	      const NumericInfo<T2> info2);
#endif
/*!
  \ingroup buildblock
  \brief Converts the \c data_in Array to \c data_out (with elements of different types) such that \c data_in == \c data_out * \c scale_factor

  \par example 
  \code
      convert_array(data_out, scale_factor, data_in);
  \endcode

  \see convert_array(scale_factor, data_in, info2) for more info
*/

template <int num_dimensions, class T1, class T2, class scaleT>
void
convert_array(Array<num_dimensions, T2>& data_out,
	      scaleT& scale_factor,
	      const Array<num_dimensions, T1>& data_in);

#if defined(ARRAY_FULL) 
#if 1
template <int num_dimensions, class T1, class T2, class scaleT>
Array<num_dimensions, T2>
convert_array_FULL(scaleT& scale_factor,
	      const Array<num_dimensions, T1>& data_in, 
	      const NumericInfo<T2> info2);
#endif
/*
   Second version:

   example :
      convert_array(data_out, scale_factor, data_in)
*/

template <int num_dimensions, class T1, class T2, class scaleT>
void
convert_array_FULL(Array<num_dimensions, T2>& data_out,
	      scaleT& scale_factor,
	      const Array<num_dimensions, T1>& data_in, 
	      const NumericInfo<T2> info2);

// #define convert_array convert_array_FULL
#endif // ARRAY_FULL

END_NAMESPACE_STIR

#endif
