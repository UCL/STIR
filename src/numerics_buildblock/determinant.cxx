//
//
/*
    Copyright (C) 2004- 2011, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup numerics
  
  \brief Implementation of stir::determinant() function for matrices
    
  \author Kris Thielemans

*/
#include "stir/Array.h"
#include "stir/numerics/determinant.h"
#include <complex>
#include "stir/Array_complex_numbers.h"

START_NAMESPACE_STIR

namespace detail
{

  template <class elemT>
  static elemT
  determinant_size3(const Array<2,elemT>& m)
  {
    const int i0 = m.get_min_index();
    const int j0 = m[i0].get_min_index();
    return
      m[i0+0][j0+0]*m[i0+1][j0+1]*m[i0+2][j0+2] +
      m[i0+0][j0+1]*m[i0+1][j0+2]*m[i0+2][j0+0] +
      m[i0+0][j0+2]*m[i0+1][j0+0]*m[i0+2][j0+1] -
      m[i0+0][j0+2]*m[i0+1][j0+1]*m[i0+2][j0+0] -
      m[i0+0][j0+1]*m[i0+1][j0+0]*m[i0+2][j0+2] -
      m[i0+0][j0+0]*m[i0+1][j0+2]*m[i0+2][j0+1];
  }

  template <class elemT>
  static elemT
  determinant_size2(const Array<2,elemT>& m)
  {
    const int i0 = m.get_min_index();
    const int j0 = m[i0].get_min_index();
    return
      m[i0+0][j0+0]*m[i0+1][j0+1] -
      m[i0+0][j0+2]*m[i0+1][j0+1];
  }

  template <class elemT>
  static elemT
  determinant_size1(const Array<2,elemT>& m)
  {
    const int i0 = m.get_min_index();
    const int j0 = m[i0].get_min_index();
    return
      m[i0][j0];
  }
}

template <class elemT>
elemT
determinant(const Array<2,elemT>& m)
{
  assert(m.is_regular());
  if (m.size() == 1)
    return detail::determinant_size1(m);
  if (m.size() == 2)
    return detail::determinant_size2(m);
  if (m.size() == 3)
    return detail::determinant_size3(m);
  error("determinant called for size larger than 3. Code in file %s needs work",
        __FILE__);
  // return to avoid compiler warning
  return 0;
}

//
// instantiations
//
template float determinant(const Array<2,float>&);
template double determinant(const Array<2,double>&);
template std::complex<float> determinant(const Array<2,std::complex<float> >&);
template std::complex<double> determinant(const Array<2,std::complex<double> >&);
END_NAMESPACE_STIR
