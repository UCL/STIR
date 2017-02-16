//
//
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
  \ingroup numerics
  
  \brief Implementation of functions for matrices
    
  \author Kris Thielemans
  \author Sanida Mustafovic

*/
#include "stir/IndexRange2D.h"
#include <complex>
#include <cmath>
# ifdef BOOST_NO_STDC_NAMESPACE
 namespace std { using ::acos; }
# endif

START_NAMESPACE_STIR


//----------------------------------------------------------------------
// some functions specific for 1D Arrays
//----------------------------------------------------------------------

template<class elemT>
inline elemT 
inner_product (const Array<1,elemT> & v1, const Array<1,elemT> &v2)
{
  assert(v1.get_index_range() == v2.get_index_range());
  elemT tmp = 0;
  typename Array<1,elemT>::const_iterator i1=v1.begin();
  typename Array<1,elemT>::const_iterator i2=v2.begin();
  for (; i1!=v1.end(); ++i1, ++i2)
    tmp += ((*i1) * (*i2));
  return tmp;
}

template<class elemT>
inline std::complex<elemT> 
inner_product (const Array<1,std::complex<elemT> > & v1, const Array<1,std::complex<elemT> > &v2)
{
  assert(v1.get_index_range() == v2.get_index_range());
  std::complex<elemT> tmp = 0;
  typename Array<1,std::complex<elemT> >::const_iterator i1=v1.begin();
  typename Array<1,std::complex<elemT> >::const_iterator i2=v2.begin();
  for (; i1!=v1.end(); ++i1, ++i2)
    tmp += (std::conj(*i1) * (*i2));
  return tmp;
}

template<class elemT>
inline double
angle (const Array<1,elemT> & v1, const Array<1,elemT> &v2)
{
  return std::acos(inner_product(v1,v2)/norm(v1)/ norm(v2));
}



//----------------------------------------------------------------------
//  functions for matrices

// matrix with vector multiplication
// first define a help function that works with both types of vectors)
namespace detail 
{
  template <class elemT, class vecT>	
  inline 
  void
  matrix_multiply_help(vecT& retval, const Array<2,elemT>& m, const vecT& vec)
  {
    assert(m.is_regular());
    if (m.size()==0)
      { return; }

    const int m_min_row = m.get_min_index();
    const int m_max_row = m.get_max_index();
    const int m_min_col = m[m_min_row].get_min_index();
    const int m_max_col = m[m_min_row].get_max_index();
    // make sure matrices are conformable for multiplication
    assert(vec.get_min_index() == m_min_col);
    assert(vec.get_max_index() == m_max_col);
    for(int i=m_min_row; i<=m_max_row; ++i)
      {
	int j=m_min_col;
	retval[i] = m[i][j]*vec[j];
	for(++j; j<=m_max_col; ++j)
	  retval[i] += m[i][j]*vec[j];
      }
  }

  template <class elemT, class vecT>
  inline
  void
  matrix_matrixT_multiply_help(vecT& retval, const Array<2,elemT>& m, const vecT& vec)
  {
      assert(m.is_regular());
      if (m.size()==0)
      { return; }

      const int m_min_row = m.get_min_index();
      const int m_max_row = m.get_max_index();
      const int m_min_col = m[m_min_row].get_min_index();
      const int m_max_col = m[m_min_row].get_max_index();
      // make sure matrices are conformable for multiplication
      assert(vec.get_min_index() == m_min_col);
      assert(vec.get_max_index() == m_max_col);

      float cov = 0.f;
      float norm_factor = 1.f / m_max_row;
      int k = m_min_row;

      for(int i=m_min_col; i<=m_max_col; ++i)
      {
          for(int j= m_min_col; j<= i; ++j)
          {
              k = m_min_row;
              cov = m[k][i] * m[k][j];
              for (++k; k<= m_max_row; ++k)
              {
                  cov += m[k][i] * m[k][j];
              }
              retval[j] += cov * norm_factor * vec[i];
              if (i!=j)
                  retval[i] += cov * norm_factor * vec[j];
          }
      }
  }
}

template <class elemT>	
inline Array<1,elemT> 
matrix_multiply(const Array<2,elemT>& m, const Array<1,elemT>& vec)
{
  Array<1,elemT> ret(m.get_min_index(), m.get_max_index());
  detail::matrix_multiply_help(ret, m, vec);
  return ret;
}

template <class elemT>
inline Array<1,elemT>
matrix_matrixT_multiply(const Array<2,elemT>& m, const Array<1,elemT>& vec)
{
  Array<1,elemT> ret(vec.size_all());
  detail::matrix_matrixT_multiply_help(ret, m, vec);
  return ret;
}

template <int num_dimensions, class elemT>	
inline BasicCoordinate<num_dimensions,elemT> 
matrix_multiply(const Array<2,elemT>& m, const BasicCoordinate<num_dimensions,elemT>& vec)
{
  BasicCoordinate<num_dimensions,elemT> ret;
  detail::matrix_multiply_help(ret, m, vec);
  return ret;
}

// matrix multiplication
template <class elemT>	
inline Array<2,elemT>
matrix_multiply(const Array<2,elemT> &m1, const Array<2,elemT>& m2) 
{
  assert(m1.is_regular());
  assert(m2.is_regular());
  if (m1.size()==0 || m2.size()==0)
    { Array<2,elemT> retval; return retval; }

  const int m1_min_row = m1.get_min_index();
  const int m1_max_row = m1.get_max_index();
  const int m2_min_row = m2.get_min_index();
  const int m2_max_row = m2.get_max_index();
  const int m2_min_col = m2[m2_min_row].get_min_index();
  const int m2_max_col = m2[m2_min_row].get_max_index();
  // make sure matrices are conformable for multiplication
  assert(m1[m1_min_row].get_min_index() == m2_min_row);
  assert(m1[m1_min_row].get_max_index() == m2_max_row);
	        
  Array<2,elemT> retval(IndexRange2D(m1_min_row, m1_max_row,
				     m2_min_col, m2_max_col));

  for (int i=m1_min_row; i<=m1_max_row; ++i)
    {
      for(int j=m2_min_col; j<=m2_max_col; ++j) 
	{
	  for(int k=m2_min_row; k<=m2_max_row; ++k)
	    retval[i][j] += m1[i][k]*m2[k][j];
	}
    }
  return retval;
}


template <class elemT>	
inline Array<2,elemT>
matrix_transpose (const Array<2,elemT>& m) 
{
  assert(m.is_regular());
  if (m.size()==0)
    { Array<2,elemT> retval; return retval; }

  const int m_min_row = m.get_min_index();
  const int m_max_row = m.get_max_index();
  const int m_min_col = m[m_min_row].get_min_index();
  const int m_max_col = m[m_min_row].get_max_index();
  Array<2,elemT> new_m(IndexRange2D(m_min_col, m_max_col,
				    m_min_row, m_max_row));
  for(int j=m_min_row; j<=m_max_row; ++j)
    for(int i=m_min_col; i<=m_max_col; ++i)
      new_m[i][j] = m[j][i];
  return new_m; 
}

template <class elemT>
inline 
Array<2,elemT>
  diagonal_matrix(const unsigned dimension, const elemT value)
{
  Array<2,elemT> m(IndexRange2D(dimension,dimension));
  for (unsigned int i=0; i<dimension; ++i)
    m[i][i]=value;
  return m;
}

template <int dimension, class elemT>
inline 
Array<2,elemT>
  diagonal_matrix(const BasicCoordinate<dimension,elemT>& values)
{
  Array<2,elemT> m(IndexRange2D(dimension,dimension));
  for (unsigned int i=0; i<dimension; ++i)
    m[i][i]=values[i+1];
  return m;
}

END_NAMESPACE_STIR
