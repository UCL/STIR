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
#ifndef __stir_numerics_max_eigenvector_H__
#define __stir_numerics_max_eigenvector_H__
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
#include "stir/more_algorithms.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

/*!
  \ingroup numerics
  \brief Compute the eigenvalue with the largest absolute value 
  and corresponding eigenvector of a matrix by using the power method.

  \param[out] max_eigenvalue will be set to the eigenvalue found
  \param[out] max_eigenvector will be set to the eigenvector found, and
    is normalised to 1 (using the l2-norm). The sign choice is
    determined by normalising the largest element in the eigenvector to 1.
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

  This will converge to the eigenvector which has
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
					    const double tolerance = .01,
					    const unsigned long max_num_iterations = 10000UL)
{
  assert(m.is_regular());
  if (m.size()==0)
    { max_eigenvalue=0; max_eigenvector=start; return Succeeded::yes; }

  const double tolerance_squared = square(tolerance);
  Array<1,elemT> current = start;
  current /= (*abs_max_element(current.begin(), current.end()));
  unsigned long remaining_num_iterations = max_num_iterations;

  double change;
  do
    {
      max_eigenvector = matrix_multiply(m, current);
      const elemT norm_factor = *abs_max_element(max_eigenvector.begin(), max_eigenvector.end());
      max_eigenvector /= norm_factor;
      change = norm_squared(max_eigenvector - current);
      current = max_eigenvector;
      --remaining_num_iterations;
    }
  while (change > tolerance_squared && remaining_num_iterations!=0);
  
  current /= norm(current);
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

/*!
  \ingroup numerics
  \brief Compute the eigenvalue with the largest absolute value 
  and corresponding eigenvector of a matrix by using the shifted power method.

  \see absolute_max_eigenvector_using_shifted_power_method().

  The current method calls the normal power method for <code>m-shift*I</code>
  and shifts the eigenvalue back to the eigenvalue for <code>m</code>.

  This method can be used to enhance the convergence rate if you know more 
  about the eigenvalues. It can also be used to find another
  eigenvalue by shifting with the maximum eigenvalue.
  */  
template <class elemT>	
inline 
Succeeded 
absolute_max_eigenvector_using_shifted_power_method(elemT& max_eigenvalue,
						    Array<1,elemT>& max_eigenvector,
						    const Array<2,elemT>& m, 
						    const Array<1,elemT>& start,
						    const elemT shift,
						    const double tolerance = .03,
						    const unsigned long max_num_iterations = 10000UL)
{
  if (m.get_min_index()!=0 || m[0].get_min_index()!=0)
    error("absolute_max_eigenvector_using_shifted_power_method:\n"
	  "  implementation needs work for indices that don't start from 0. sorry");

  Succeeded success =
    absolute_max_eigenvector_using_power_method(max_eigenvalue,
						max_eigenvector,
						// sadly need to explicitly convert result of subtraction back to Array
						Array<2,elemT>(m - diagonal_matrix(m.size(), shift)), 
						start,
						tolerance,
						max_num_iterations);
  max_eigenvalue += shift;
  return success;
}


/*!
  \ingroup numerics
  \brief Compute the eigenvalue with the largest value 
  and corresponding eigenvector of a matrix by using the power method.

  \warning This assumes that all eigenvalues are real.

  \see absolute_max_eigenvector_using_shifted_power_method().

  \param[in] m is the input matrix, which has to be real-symmetric


  This will attempt to find the eigenvector which has
  the largest eigenvalue. The method fails when the matrix
  has a negative eigenvalue of the same magnitude as the 
  largest eigenvalue.

  \todo the algorithm would work with hermitian matrices, but the code needs one small adjustment.
*/  
template <class elemT>	
inline 
Succeeded 
max_eigenvector_using_power_method(elemT& max_eigenvalue,
				   Array<1,elemT>& max_eigenvector,
				   const Array<2,elemT>& m, 
				   const Array<1,elemT>& start,
				   const double tolerance = .03,
				   const unsigned long max_num_iterations = 10000UL)
{

  Succeeded success =
    absolute_max_eigenvector_using_power_method(max_eigenvalue,
						max_eigenvector,
						m,
						start,
						tolerance,
						max_num_iterations);
  if (success == Succeeded::no)
    return Succeeded::no;

  if (max_eigenvalue>=0) // TODO would need to take real value for complex case
    return Succeeded::yes;

  // we found a negative eigenvalue
  // try again with a shift equal to the previously found max_eigenvalue
  // this shift will effectively put that eigenvalue to 0 during the
  // power method iterations
  // also it will make all eigenvalues positive (as we subtract the 
  // smallest negative eigenvalue)
  success =
    absolute_max_eigenvector_using_shifted_power_method(max_eigenvalue,
							max_eigenvector,
							m,
							start,
							max_eigenvalue,
							tolerance,
							max_num_iterations);
  if (success == Succeeded::no)
    return Succeeded::no;

  assert(max_eigenvalue>=0);
  return Succeeded::yes;
}

END_NAMESPACE_STIR

#endif
