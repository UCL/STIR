//
// $Id$: $Date$
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

  \date    $Date$
  \version $Revision$

*/

// TODO enable FULL 

#include "NumericInfo.h"
#include "Array.h"

START_NAMESPACE_TOMO

/*!
   \brief A function that returns a new Array with elements of type \c T2 such that \c data_in == \c data_out * \c scale_factor

   example :
      data_out = convert_array(scale_factor, data_in, NumericInfo<T2>())

   \param scale_factor 
          a reference to a (float or double) variable which will be
	  set to the scale factor such that (ignoring types)
	     data_in == data_out * scale_factor
	  If scale_factor is initialised to 0, the maximum range of T2
	  is used. If scale_factor != 0, convert_array attempts to use the
	  given scale_factor, unless the T2 range doesn't fit.
	  In that case, the same scale_factor is used as in the 0 case.
   
   \param data_in 
          some Array object, elements are of some numeric type \c T1
   \param  2nd parameter :
          T2 is the desired output type

   \return 
      data_out :
          an Array object whose elements are of numeric type T2.

   When the output type is integer, rounding is used.

   Note that there is an effective threshold at 0 currently (i.e. negative
   numbers are cut out) when T2 is an unsigned type.
*/
#if 1
template <int num_dimensions, class T1, class T2, class scaleT>
Array<num_dimensions, T2>
convert_array(scaleT& scale_factor,
	      const Array<num_dimensions, T1>& data_in, 
	      const NumericInfo<T2> info2);
#endif
/*!

  \brief Converts the \c data_in Array to \c data_out (with elements of different types) such that \c data_in == \c data_out * \c scale_factor

  TODOdoc more 

  \par example 
  \code
      convert_array(data_out, scale_factor, data_in);
  \endcode
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

END_NAMESPACE_TOMO

#endif
