// $Id$
/*!

  \file
  \ingroup buildblock

  \brief This include file provides some additional functionality for Array objects.

  \author Kris Thielemans (loosely based on some earlier work by Darren Hague)
  \author PARAPET project

  $Date$
  $Revision$

  <ul>
  <li>
   functions which work on all Array objects, and which change every element of the
   array: 
   <ul>
     <li> in_place_log, in_place_exp (these work only well when elements are float or double)
     <li>in_place_abs
     <li>in_place_apply_function
     <li>in_place_apply_array_function_on_1st_index
     <li>in_place_apply_array_functions_on_each_index
   </ul>
   All these functions return a reference to the (modified) array
   <li>
   Analoguous functions that take out_array and in_array
   <ul>
      <li>apply_array_function_on_1st_index
      <li>apply_array_functions_on_each_index
   </ul>
   <li>
   functions specific to Array<1,elemT>:
   <ul>
     <li>inner_product, norm, angle
   </ul>
   <li>
   functions specific to Array<2,elemT>:
   <ul>
     <li>matrix_transpose, matrix_multiply (both currently disabled as they need testing)
   </ul>
   </ul>

   \warning Compilers without partial specialisation of templates are
   catered for by explicit instantiations. If you need it for any other
   types, you'd have to add them by hand.
 */
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

/* History:

  KT 21/05/2001
  added in_place_apply_array_function_on_1st_index, 
  in_place_apply_array_function_on_each_index

  KT 06/12/2001
  added apply_array_function_on_1st_index, 
  apply_array_function_on_each_index
*/

#ifndef __stir_ArrayFunction_H__
#define __stir_ArrayFunction_H__
  
#include "stir/Array.h"
#include "stir/shared_ptr.h"

#include <cmath>

#include "stir/ArrayFunctionObject.h"

START_NAMESPACE_STIR

//----------------------------------------------------------------------
// element wise and in place numeric functions
//----------------------------------------------------------------------

//! Replace elements by their logarithm, 1D version
#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <class elemT>
inline Array<1,elemT>&
in_place_log(Array<1,elemT>& v);
#else
inline Array<1,float>& 
in_place_log(Array<1,float>& v);
#endif


//! apply log to each element of the multi-dimensional array
template <int num_dimensions, class elemT>
inline Array<num_dimensions, elemT>& 
in_place_log(Array<num_dimensions, elemT>& v);

//! Replace elements by their exponentiation, 1D version
#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <class elemT>
inline Array<1,elemT>& 
in_place_exp(Array<1,elemT>& v);
#else
inline Array<1,float>& 
in_place_exp(Array<1,float>& v);
#endif

//! apply exp to each element of the multi-dimensional array
template <int num_dimensions, class elemT>
inline Array<num_dimensions, elemT>& 
in_place_exp(Array<num_dimensions, elemT>& v);

//! Replace elements by their absolute value, 1D version
#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <class elemT>
inline Array<1,elemT>& 
in_place_abs(Array<1,elemT>& v);
#else
inline Array<1,float>& 
in_place_abs(Array<1,float>& v);
#endif

//! store absolute value of each element of the multi-dimensional array
template <int num_dimensions, class elemT>
inline Array<num_dimensions, elemT>& 
in_place_abs(Array<num_dimensions, elemT>& v);

// this generic function does not seem to work of f is an overloaded function
//! apply any function(object) to each element of the 1-dimensional array
#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <class elemT, class FUNCTION>
inline Array<1,elemT>& 
in_place_apply_function(Array<1,elemT>& v, FUNCTION f);
#else
inline Array<1,float>& 
in_place_apply_function(Array<1,float>& v, float (*f)(float));
#endif


//! apply any function(object) to each element of the multi-dimensional array
/*! each element will be replaced by 
    \code
    elem = f(elem);
    \endcode
*/
template <int num_dimensions, class elemT, class FUNCTION>
inline Array<num_dimensions, elemT>& 
in_place_apply_function(Array<num_dimensions, elemT>& v, FUNCTION f);

//! Apply a function object on all possible 1d arrays extracted by keeping all indices fixed, except the first one
/*!
  For the 2d case, this amounts to applying a function on all columns 
  of the matrix.

  For a 3d case, the following pseudo-code illustrates what happens.
  \code
  for all i2,i3
  {
    for all i
    {
      a[i] = array[i][i2][i3];
    }
    (*f)(a);
    for all i
    {
      array[i][i2][i3] = a[i];
    }
  }
  \endcode

  \warning The array has to be regular.
  \todo Add a 1D specialisation as the current implementation would be really 
  inefficient in this case.
  \todo Add a specialisation such that this function would handle function 
  objects and (smart) pointers to function objects. At the moment, it's only the latter.
*/
template <int num_dim, typename elemT, typename FunctionObjectPtr> 
inline void
in_place_apply_array_function_on_1st_index(Array<num_dim, elemT>& array, FunctionObjectPtr f);


/*! As above, but storing results in a different array.
  \warning Both in_array and out_array have to have regular ranges. Moreover, they have to 
  have matching ranges except for the outermost level. The (binary) function is applied as
  \code 
    (*f)(out_array1d, in_array1d)
  \endcode
  \a f should not modify the index range of the output argument.
  */
template <int num_dim, typename elemT, typename FunctionObjectPtr> 
inline void
apply_array_function_on_1st_index(Array<num_dim, elemT>& out_array, 
                                  const Array<num_dim, elemT>& in_array, 
                                  FunctionObjectPtr f);

/* local #define used for 2 purposes:
   - in partial template specialisation that uses ArrayFunctionObject types
   - in complete template specialisation for deficient compilers

   This is done with a #define to keep code reasonably clean. Also, it allows
   adjusting the type according to what you need.
   Still, it's terribly ugly. Sorry.

  Note that you shouldn't/cannot use this define outside of this include file 
  (and its .inl) partner.

  Ideally, the code should be rewritten to work with any kind of (smart) ptr. TODO
*/
#if !defined(__GNUC__) && !defined(_MSC_VER)
#define ActualFunctionObjectPtrIter \
  VectorWithOffset< shared_ptr<ArrayFunctionObject<1,elemT> > >::const_iterator  
#else
/*
  Puzzlingly, although the code is actually  called with iterators of the type above,
  gcc 3.0 (and others?) gets confused and refuses to compile the 
  partial template specialisation (it says it's ambiguous).
  VC also refuses to compile it.
  A work-around is to use the following type
*/
#define ActualFunctionObjectPtrIter shared_ptr<ArrayFunctionObject<1,elemT> > const* 
#endif

#ifdef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
// silly business for deficient compilers (including VC 6.0)
#define elemT float
#define FunctionObjectPtrIter ActualFunctionObjectPtrIter
#endif


//! Apply a sequence of 1d array-function objects on every dimension of the input array
/*!
  The sequence of function object pointers is specified by iterators. There must be
  num_dim function objects in the sequence, i.e. stop-start==num_dim.

  The n-th function object (**(start+n)) is applied on the n-th index of the
  array. So, (*start) is applied using
  in_place_apply_array_function_on_1st_index(array, *start).
  Similarly, (*(start+1) is applied using
  in_place_apply_array_function_on_1st_index(array[i], *(start+1))
  for every i. And so on.
  \todo Add a specialisation such that this function would handle function 
  objects and (smart) pointers to function objects. At the moment, it's only the latter.
*/
// TODO add specialisation that uses ArrayFunctionObject::is_trivial
#ifdef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <int num_dim>
#else
template <int num_dim, typename elemT, typename FunctionObjectPtrIter> 
#endif
inline void 
in_place_apply_array_functions_on_each_index(Array<num_dim, elemT>& array, 
                                             FunctionObjectPtrIter start, 
                                             FunctionObjectPtrIter stop);

//! 1d specialisation of the above. 
#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <typename elemT, typename FunctionObjectPtrIter> 
#endif
inline void 
in_place_apply_array_functions_on_each_index(Array<1, elemT>& array, 
                                             FunctionObjectPtrIter start, 
                                             FunctionObjectPtrIter stop);



#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
//! Apply a sequence of 1d array-function objects on every dimension of the input array, store in output array
/*!
  The sequence of function object pointers is specified by iterators. There must be
  num_dim function objects in the sequence, i.e. stop-start==num_dim.

  The n-th function object (**(start+n)) is applied on the n-th indices of the
  arrays. So, (*start) is applied using
  \code
  apply_array_function_on_1st_index(out_array, in_array, *start).
  \endcode
  and so on.
  \todo Add a specialisation such that this function would handle iterators of function 
  objects and (smart) pointers to function objects. At the moment, it's only the latter.
*/
template <int num_dim, typename elemT, typename FunctionObjectPtrIter> 
inline void 
apply_array_functions_on_each_index(Array<num_dim, elemT>& out_array, 
                                    const Array<num_dim, elemT>& in_array, 
                                    FunctionObjectPtrIter start, 
                                    FunctionObjectPtrIter stop);
#endif

//! Apply a sequence of 1d array-function objects of a specific type on every dimension of the input array, store in output array
/*!
  This function uses optimisations possible because ArrayFunctionObject gives information
  on sizes etc.
  \todo Modify such that this function would handle function 
  objects and (smart) pointers to ArrayFunctionObject objects. At the moment, it's only the latter.
*/
#ifdef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <int num_dim>
#else
template <int num_dim, typename elemT> 
#endif
inline void 
apply_array_functions_on_each_index(Array<num_dim, elemT>& out_array, 
                                    const Array<num_dim, elemT>& in_array, 
                                    ActualFunctionObjectPtrIter start, 
                                    ActualFunctionObjectPtrIter stop);

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
//! 1d specialisation of above
// has to be here to get general 1D specialisation to compile
template <typename elemT>
#endif
inline void 
apply_array_functions_on_each_index(Array<1, elemT>& out_array, 
                                    const Array<1, elemT>& in_array, 
                                    ActualFunctionObjectPtrIter start, 
                                    ActualFunctionObjectPtrIter stop);

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <typename elemT, typename FunctionObjectPtrIter> 
//! 1d specialisation for general function objects
inline void 
apply_array_functions_on_each_index(Array<1, elemT>& out_array, 
                                    const Array<1, elemT>& in_array, 
                                    FunctionObjectPtrIter start, FunctionObjectPtrIter stop);
#endif
                                    
#ifdef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
#undef elemT
#undef FunctionObjectPtrIter
#endif


//----------------------------------------------------------------------
// some functions specific for 1D Arrays
//----------------------------------------------------------------------

//! Inner product of 2 1D arrays
/*! This returns the sum of multiplication of elements of \a v1 and \a v2.
    \warning Implementation is not appropriate for complex numbers.
 */
template<class elemT>
inline elemT 
inner_product (const Array<1,elemT> & v1, const Array<1,elemT> &v2);

//! l2 norm of a 1D array
/*! This returns the sqrt of the sum of the square elements of \a v1.
    \warning Implementation is not appropriate for complex numbers.
 */
template<class elemT>
inline double 
norm (const Array<1,elemT> & v1);

//! angle between 2 1D arrays
/*! This is not appropriate for complex numbers.
 */
template<class elemT>
inline double
angle (const Array<1,elemT> & v1, const Array<1,elemT> &v2);


END_NAMESPACE_STIR

#include "stir/ArrayFunction.inl"
#undef ActualFunctionObjectPtrIter

#endif

