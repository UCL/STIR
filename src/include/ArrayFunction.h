// $Id$: $Date$
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
   tensor: 
   in_place_log, in_place_exp (these work only well when elements are float or double)
   in_place_abs
   in_place_apply_function<br>
   All these functions return a reference to the (modified) array
   <li>
   functions specific to Array<1,elemT>:
   inner_product, norm, angle
   <li>
   functions specific to Array<2,elemT>:
   matrix_transpose, matrix_multiply
   </ul>
 
 */
#ifndef __ArrayFunction_H__
#define __ArrayFunction_H__
  
#include "Array.h"

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

// KT 28/10/98 corrected F->f and recursion
template <int num_dimensions, class elemT, class FUNCTION>
inline Array<num_dimensions, elemT>& 
in_place_apply_function(Array<num_dimensions, elemT>& v, FUNCTION f)  
{	
  for(int i=v.get_min_index(); i<=v.get_max_index(); i++)
    in_place_apply_function(v[i], f); 
  return v; 
}

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

