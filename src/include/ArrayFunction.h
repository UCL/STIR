// $Id$
/*!

  \file
  \ingroup buildblock

  \brief This include file provides some additional functionality for Array objects.

  \author Kris Thielemans (loosely based on some earlier work by Darren Hague)
  \author PARAPET project

  \date $Date$

  \version $Revision$

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
   functions specific to Array<1,elemT>:
   <ul>
   <li>inner_product, norm, angle
   </ul>
   <li>
   functions specific to Array<2,elemT>:
   <ul>
   <li>matrix_transpose, matrix_multiply
   </ul>
   </ul>

   \warning Compilers without partial specialisation of templates are
   catered for by explcit instantiations. If you need it for any other
   types, you'd have to add them by hand.
 */
#ifndef __ArrayFunction_H__
#define __ArrayFunction_H__
  
#include "Array.h"
#include "shared_ptr.h"

#include <cmath>

START_NAMESPACE_TOMO

//----------------------------------------------------------------------
// element wise and in place numeric functions
//----------------------------------------------------------------------

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <class elemT>
inline Array<1,elemT>&
in_place_log(Array<1,elemT>& v)  
{	
  for(int i=v.get_min_index(); i<=v.get_max_index(); i++)
    v[i] = log(v[i]); 
  return v; 
}

#else
inline Array<1,float>& 
in_place_log(Array<1,float>& v)  
{	
  for(int i=v.get_min_index(); i<=v.get_max_index(); i++)
    v[i] = log(v[i]); 
  return v; 
}
#endif


//! apply log to each element of the multi-dimensional array
template <int num_dimensions, class elemT>
inline Array<num_dimensions, elemT>& 
in_place_log(Array<num_dimensions, elemT>& v)  
{	
  for(int i=v.get_min_index(); i<=v.get_max_index(); i++)
    in_place_log(v[i]); 
  return v; 
}

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <class elemT>
inline Array<1,elemT>& 
in_place_exp(Array<1,elemT>& v)  
{	
  for(int i=v.get_min_index(); i<=v.get_max_index(); i++)
    v[i] = exp(v[i]); 
  return v; 
}
#else
inline Array<1,float>& 
in_place_exp(Array<1,float>& v)  
{	
  for(int i=v.get_min_index(); i<=v.get_max_index(); i++)
    v[i] = exp(v[i]); 
  return v; 
}
#endif

//! apply exp to each element of the multi-dimensional array
template <int num_dimensions, class elemT>
inline Array<num_dimensions, elemT>& 
in_place_exp(Array<num_dimensions, elemT>& v)  
{	
  for(int i=v.get_min_index(); i<=v.get_max_index(); i++)
    in_place_exp(v[i]); 
  return v; 
}

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <class elemT>
inline Array<1,elemT>& 
in_place_abs(Array<1,elemT>& v)  
{	
  for(int i=v.get_min_index(); i<=v.get_max_index(); i++)
    if (v[i] < 0)
      v[i] = -v[i];
  return v; 
}
#else
inline Array<1,float>& 
in_place_abs(Array<1,float>& v)  
{	
  for(int i=v.get_min_index(); i<=v.get_max_index(); i++)
    if (v[i] < 0)
      v[i] = -v[i];
  return v; 
}
#endif

//! store absolute value of each element of the multi-dimensional array
template <int num_dimensions, class elemT>
inline Array<num_dimensions, elemT>& 
in_place_abs(Array<num_dimensions, elemT>& v)  
{	
  for(int i=v.get_min_index(); i<=v.get_max_index(); i++)
    in_place_abs(v[i]); 
  return v; 
}


// this generic function does not seem to work of f is an overloaded function
#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <class elemT, class FUNCTION>
inline Array<1,elemT>& 
in_place_apply_function(Array<1,elemT>& v, FUNCTION f)  
{	
  for(int i=v.get_min_index(); i<=v.get_max_index(); i++)
    v[i] = f(v[i]); 
  return v; 
}
#else
inline Array<1,float>& 
in_place_apply_function(Array<1,float>& v, float (*f)(float))  
{	
  for(int i=v.get_min_index(); i<=v.get_max_index(); i++)
    v[i] = f(v[i]); 
  return v; 
}
#endif


//! apply any function(object) to each element of the multi-dimensional array
/*! each element will be replaced by 
    \code
    elem = f(elem);
    \endcode
*/
template <int num_dimensions, class elemT, class FUNCTION>
inline Array<num_dimensions, elemT>& 
in_place_apply_function(Array<num_dimensions, elemT>& v, FUNCTION f)  
{	
  for(int i=v.get_min_index(); i<=v.get_max_index(); i++)
    in_place_apply_function(v[i], f); 
  return v; 
}

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
    f(a);
  }
  \endcode
*/
template <int num_dim, typename elemT, typename FunctionObject> 
void
in_place_apply_array_function_on_1st_index(Array<num_dim, elemT>& array, FunctionObject f)
{
  assert(array.is_regular());
  const int outer_min_index = array.get_min_index();
  const int outer_max_index = array.get_max_index();

  // construct a vector with a full_iterator for every array[i]
  VectorWithOffset< Array<num_dim-1, elemT>::full_iterator > 
    full_iterators (outer_min_index, outer_max_index);  
  for (int i=outer_min_index; i<=outer_max_index; ++i)
    full_iterators[i] = array[i].begin_all();
  
  // allocate 1d array
  Array<1, elemT> array1d (outer_min_index, outer_max_index);

  while (full_iterators[outer_min_index] != array[outer_min_index].end_all())
  {
    // copy elements into 1d array
    for (int i=outer_min_index; i<=outer_max_index; ++i)    
      array1d[i] = *full_iterators[i];
    
    // apply function
    (*f)(array1d);
    
    // put results back
    for (int i=outer_min_index; i<=outer_max_index; ++i)
      *full_iterators[i] = array1d[i];
    
    // now increment full_iterators to do next index
    for (int i=outer_min_index; i<=outer_max_index; ++i)
      ++full_iterators[i];
  }
    
}

//! Apply a sequence of 1d array-function objects on every dimension of the input array
/*!
  The sequence of function object pointers is specified by iterators. There must be
  num_dim function objects in the sequence, i.e. stop-start==num_dim.

  The n-th function object (*(start+n)) is applied on the n-th index of the
  array. So, (*start) is applied using
  in_place_apply_array_function_on_1st_index(array, *start).
  Similarly, (*(start+1) is applied using
  in_place_apply_array_function_on_1st_index(array[i], *(start+1))
  for every i. And so on.
*/
// TODO add specialisation that uses ArrayFunctionObject::is_trivial
template <int num_dim, typename elemT, typename FunctionObjectIter> 
void 
in_place_apply_array_functions_on_each_index(Array<num_dim, elemT>& array, 
                                             FunctionObjectIter start, 
                                             FunctionObjectIter stop)
{
  assert(start+num_dim == stop);
  assert(num_dim > 1);
  in_place_apply_array_function_on_1st_index(array, *start);

  ++start;
  for (Array<num_dim, elemT>::iterator restiter = array.begin(); restiter != array.end(); ++restiter)
    in_place_apply_array_functions_on_each_index(*restiter, start, stop);
}



typedef void (*floatarrayfunctionptr)(Array<1,float>&);

//! 1d specialisation. 
/*!
  \warning 
  Currently uses (**start)(array), while preferably it would use (*start)(array). However, this doesn't compile when the function objects are part of a 
 std::vector< shared_ptr<ArrayFunctionObject<1,elemT> > >.
 (TODO)
*/
#ifdef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
// silly business for deficient compilers (including VC 6.0)
#define elemT float
#define FunctionObjectIter std::vector< shared_ptr<ArrayFunctionObject<1,elemT> > >::const_iterator 
#else
template <typename elemT, typename FunctionObjectIter> 
#endif
void in_place_apply_array_functions_on_each_index(Array<1, elemT>& array, FunctionObjectIter start, FunctionObjectIter stop)
{
  assert(start != stop);
  (**start)(array);
  assert (++start == stop);
}

#ifdef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
#undef elemT
#undef FunctionObjectIter
#endif


//----------------------------------------------------------------------
// some functions specific for 1D Arrays
//----------------------------------------------------------------------

template<class elemT>
inline elemT 
inner_product (const Array<1,elemT> & v1, const Array<1,elemT> &v2)
{
  return (v1 * v2).sum(); 
}

template<class elemT>
double 
norm (const Array<1,elemT> & v1)
{
  return sqrt((double)inner_product(v1,v1));
}

#if 0
// something like this for complex elemTs, but we don't use them
inline double 
norm (const Array<1,complex<double>> & v1)
{
  return sqrt(inner_product(v1,conjugate(v1)).re());

}
#endif

template<class elemT>
inline elemT 
angle (const Array<1,elemT> & v1, const Array<1,elemT> &v2)
{
  return acos(inner_product(v1,v2)/norm(v1)/ norm(v2));
}


//----------------------------------------------------------------------
// some matrix manipulations
//----------------------------------------------------------------------
//TODO to be tested
// KT 29/10/98 as they need to be tested, I disabled them for the moment
#if 0
template <class elemT>	
inline Array<2,elemT>
matrix_transpose (const Array<2,elemT>& matrix) 
{
  Array<2,elemT> new_matrix(matrix.get_min_index2(), matrix.get_max_index2(),
			      matrix.get_min_index1(), matrix.get_max_index1());
  for(int j=matrix.get_min_index2(); j<=matrix.get_max_index2(); j++)
    for(int i=matrix.get_min_index1(); i<=matrix.get_max_index1(); i++) {
      new_matrix[i][j] = matrix[j][i];
    }
  return new_matrix; 
}

template <class elemT>	
inline Array<2,elemT>
matrix_multiply(const Array<2,elemT> &m1, const Array<2,elemT>& m2) 
{

  //KT TODO what to do with growing ? 
  // for the moment require exact matches on sizes as in assert() below

  // make sure matrices are conformable for multiplication
  assert(m1.get_width() == m2.get_height() && m1.get_w_min() == m2.get_h_min());  
	        
  Array<2,elemT> retval(m1.get_h_min(), m1.get_h_max(), 
			  m2.get_w_min(), m2.get_w_max());
#if 0
  // original version, works only when member of Array<2,*>
  for (int i=0; i<height; i++)
    {
      register elemT *numi = &(retval.num[i][0]);// use these temp vars for efficiency
      register elemT numik;
      for(int k=0; k<width; k++)
	{
	  numik = num[i][k];
	  const Array<1,elemT> ivtk = iv[k];
	  for(register int j=0; j<iv.width; j++) 
	    numi[j] += (numik*ivtk[j]);
	}
    }
#endif
  // new version by KT
  for (int i=m1.get_h_min(); i<=m1.get_h_max(); i++)
    {
      // use these temp vars for efficiency (is this necessary ?)
      const Array<1,elemT>& m1_row_i = m1[i];
      Array<1,elemT>& new_row_i = retval[i];
      for(int k=m1.get_w_min(); k<=m1.get_w_max(); k++)
	{
	  elemT m1_ik = m1_row_i[k];
	  const Array<1,elemT>& m2_row_k = m2[k];
	  for(int j=m2.get_w_min(); j<=m2.get_w_max(); j++) 
	    new_row_i[j] += (m1_ik*m2_row_k[j]);
	}
    }
  return retval;
}
#endif // 0 (2D routines)

END_NAMESPACE_TOMO

#endif

