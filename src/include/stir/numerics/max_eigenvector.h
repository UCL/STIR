//
// $Id$
//
/*
    Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
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
#ifndef __stir_max_eigenvector_H__
#define __stir_max_eigenvector_H__
/*!
  \file
  \ingroup numerics
  
  \brief Declaration of functions for computing eigenvectors
    
  \author Kris Thielemans
  \author Sanida Mustafovic

  $Date$
  $Revision$
*/
#include "stir/numerics/MatrixFunction.h"
#include "stir/numerics/norm.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

/*!
  \ingroup numerics
  \brief Compute the eigenvalue with the largest absolute value 
  and corresponding eigenvector of a matrix.

  \param[out] max_eigenvalue will be set to the eigenvalue found
  \param[out] max_eigenvector will be set to the eigenvector found, and
    is normalised to 1 (using the l2-norm).
  \param[in] m is the input matrix
  \param[in] start is a starting vector for the iterations
  \param[in] tolerance determines when iterations can stop
  \param[in]  max_num_iterations is used to prevent an infinite loop

  \return Succeeded::yes if \a max_num_iterations was not reached.
  However, you probably want to check if the
  norm of the difference between  <code>m.max_eigenvector</code>
  and <code> max_eigenvalue*max_eigenvector</code> is small
  (compared to max_eigenvalue).  


  Computation uses the <i>power</a> method, see for instance
  http://www.maths.lth.se/na/courses/FMN050/FMN050-05/eigenE.pdf.

  The method consists in computing 
  \f[v^{n+1}=m.v^{n}\f]
  \f[v^{n+1}/=\mathrm{norm}(v^{n+1})\f]
 with
  \f$ v^{0}=\mathrm{start} \f$
  for \f$ n\f$ big enough such that 
  \f$ \mathrm{norm}(v^{n+1}-v^{n})\f$ becomes smaller 
  than \a tolerance. The eigenvalue is then computed using the Rayleigh
  quotient 
  \f[v.m.v \over v.v \f]

  This will converge to the eigenvector with eigenvalue which has
  the largest absolute eigenvalue. The method fails when the matrix
  has more than 1 largest absolute eigenvalue
  (e.g. with opposite sign).

*/  
template <class elemT>	
inline 
Succeeded 
absolute_max_eigenvector_using_power_method(elemT& max_eigenvalue,
					    Array<1,elemT>& max_eigenvector,
					    const Array<2,elemT>& m, 
					    const Array<1,elemT>& start,
					    const double tolerance = .03,
					    const unsigned long max_num_iterations = 10000000L)
{
  assert(m.is_regular());
  if (m.size()==0)
    { max_eigenvalue=0; max_eigenvector=start; return; }

#ifndef NDEBUG
  const int m_min = m.get_min_index();
  const int m_max = m.get_max_index();
  assert(m_min_row = m[m_min].get_min_index());
  assert(m_max_row = m[m_min].get_max_index());
  assert(start.get_min_index() == m_min);
  assert(start.get_max_index() == m_max);
#endif

  Array<1,elemT> current = start;
  current /= norm(current);
  unsigned long remaining_num_iterations = max_num_iterations;

  do
    {
      max_eigenvector = matrix_multiply(m, current);
      max_eigenvector /= norm(max_eigenvector);
      const double change = norm(max_eigenvector - current);
      current = max_eigenvector;
      --remaining_num_iterations;
    }
  while (change > tolerance && remaining_num_iterations!=0);
  
  max_eigenvector = matrix_multiply(m, current);
  // compute eigenvalue using Rayleigh quotient
  max_eigenvalue = 
    inner_product(current,max_eigenvector) /
    norm_squared(current);
  max_eigenvector /= norm(max_eigenvector);

  return remaining_num_iterations ==0 ? Succeeded::no : Succeeded::yes;

  /*norm( max_eigenvector*max_eigenvalue -
	       matrix_multiply(m, max_eigenvector));*/
}
END_NAMESPACE_STIR
