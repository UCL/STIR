//
// $Id$
//

/*
  \file
  \ingroup buildblock

  \brief Implementation for linear_regression() function

  \author Kris Thielemans  
  \author PARAPET project

  $Date$
  $Revision$
*/  
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    This software is distributed under the terms 
    of the GNU Lesser General  Public Licence (LGPL)
    See STIR/LICENSE.txt for details
*/

#include "stir/linear_regression.h"

START_NAMESPACE_STIR

namespace detail {

  // see linear_regression.inl for more info


template <class Value>
void linear_regression_compute_fit_from_S(Value& constant, Value& scale,
					  Value& chi_square,
					  Value& variance_of_constant,
					  Value& variance_of_scale,
					  Value& covariance_of_constant_with_scale,
					  const double S,
					  const double Sx,  
					  const double Sy,
					  const double Syy,
					  const double Stt,
					  const double Sty,
					  const unsigned data_size,
					  const bool use_estimated_variance
                       )
{

  
  
  scale = static_cast<Value>(Sty / Stt);
  constant = static_cast<Value>((Sy - Sx * scale) / S);
  variance_of_scale = static_cast<Value>(1/ Stt);
  variance_of_constant = static_cast<Value>((1 + square(Sx)/(S*Stt))/S);
  covariance_of_constant_with_scale = static_cast<Value>(- Sx / (S*Stt));
  
  // Compute chi_square, i.e.
  // sum_i weights[i]*(measured_data[i] - (constant+scale*coordinates[i]))^2
  // It turns out this can be done using previously calculated constants,
  // and just 1 extra variable : Syy

  // Proving the next identity is a tough algebraic calculation
  // (performed by KT in Mathematica)
  chi_square = static_cast<Value>(Syy - square(scale)*Stt - square(Sy)/S);

  if (chi_square<0)
    {
      warning("linear_regression found negative chi_square %g.\n"
	      "This is probably just because of rounding errors and indicates "
	      "a small error compared to the data. I will set it to 0 to avoid "
	      "problems with sqrt(chi_square).",
	      chi_square);
      chi_square=0;
    }
  if (use_estimated_variance==true)
  {
    const Value estimated_variance = 
      static_cast<Value>(chi_square/(data_size - 2));
    variance_of_scale *=estimated_variance;
    variance_of_constant *=estimated_variance;
    covariance_of_constant_with_scale *=estimated_variance;
  }
}

// instantiations
template void 
linear_regression_compute_fit_from_S<>(float& constant, float& scale,
				       float& chi_square,
				       float& variance_of_constant,
				       float& variance_of_scale,
				       float& covariance_of_constant_with_scale,
				       const double S,
				       const double Sx,  
				       const double Sy,
				       const double Syy,
				       const double Stt,
				       const double Sty,
				       const unsigned data_size,
				       const bool use_estimated_variance
				       );

template void 
linear_regression_compute_fit_from_S<>(double& constant, double& scale,
				       double& chi_square,
				       double& variance_of_constant,
				       double& variance_of_scale,
				       double& covariance_of_constant_with_scale,
				       const double S,
				       const double Sx,  
				       const double Sy,
				       const double Syy,
				       const double Stt,
				       const double Sty,
				       const unsigned data_size,
				       const bool use_estimated_variance
				       );

} // end namespace detail

END_NAMESPACE_STIR
