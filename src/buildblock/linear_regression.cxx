//
// $Id$: $Date$
//

/*
  Linear regression function
  
  Kris Thielemans, 08/12/1999
*/


#include "pet_common.h"
#include "linear_regression.h"


template <class Value, class DataType, class CoordinatesType>
void linear_regression(Value& constant, Value& scale,
		       Value& chi_square,
		       Value& variance_of_constant,
		       Value& variance_of_scale,
		       Value& covariance_of_constant_with_scale,
		       const VectorWithOffset<DataType>& measured_data,
		       const VectorWithOffset<CoordinatesType>& coordinates,
		       const VectorWithOffset<float>& weights,
		       const bool use_estimated_variance
                       )
{

  assert(measured_data.get_min_index() == coordinates.get_min_index());
  assert(measured_data.get_min_index() == weights.get_min_index());
  assert(measured_data.get_max_index() == coordinates.get_max_index());
  assert(measured_data.get_max_index() == weights.get_max_index());
  
  
  /* 
     Use Numerical Recipes formulas for numerical stability

     Correspondence with their notation:
     y_i = measured_data[i]
     x_i = coordinates[i]
     sigma_i = 1/sqrt(weights[i])
      
     t_i = sqrt(weights[i]) (x[i] - Sx/S) 
      
     This last one is computed here in terms of 
     wti = (x[i] - Sx/S)

     Further notation:
     S = sum of the weights
     Sx = weights . coordinates
     Sy = weights . measured
     Sty = weights . (wt*measured)
     etc.
 
  */
  
  Value Sy = 0;
  Value Sx = 0;  
  Value S = 0;
  
  for (int i=measured_data.get_min_index(); i <= measured_data.get_max_index(); i++)
  {
    S += weights[i];
    Sx += weights[i]*coordinates[i];
    Sy += weights[i]*measured_data[i];
  }

  Value Stt = 0;
  Value Sty = 0;
  
  for (int i=measured_data.get_min_index(); i <= measured_data.get_max_index(); i++)
  {
    const Value wti = (coordinates[i] - Sx/S);
    Stt   += weights[i] * wti * wti;
    Sty += weights[i] * wti * measured_data[i];
  }
  
  scale = Sty / Stt;
  constant = (Sy - Sx * scale) / S;
  variance_of_scale = 1/ Stt;
  variance_of_constant = (1 + square(Sx)/(S*Stt))/S;
  covariance_of_constant_with_scale = - Sx / (S*Stt);
  
  // Compute chi_square, i.e.
  // sum_i weights[i]*(measured_data[i] - (constant+scale*coordinates[i]))^2
  // It turns out this can be done using previously calculated constants,
  // and just 1 extra variable : Syy
  Value Syy = 0;
  for (int i=measured_data.get_min_index(); i <= measured_data.get_max_index(); i++)
  {
    Syy += weights[i] * square(measured_data[i]);
  }

  // Proving the next identity is a tough algebraic calculation
  // (performed by KT in Mathematica)
  chi_square = Syy - square(scale)*Stt - square(Sy)/S;
  
  if (use_estimated_variance==true)
  {
    const Value estimated_variance = 
      chi_square/(measured_data.get_length() - 2);
    variance_of_scale *=estimated_variance;
    variance_of_constant *=estimated_variance;
    covariance_of_constant_with_scale *=estimated_variance;
  }
}

// instantiations
template 
void linear_regression<>(float& constant, float& scale,
		       float& chi_square,
		       float& variance_of_constant,
		       float& variance_of_scale,
		       float& covariance_of_constant_with_scale,
		       const VectorWithOffset<float>& measured_data,
		       const VectorWithOffset<float>& coordinates,
		       const VectorWithOffset<float>& weights,
		       const bool use_estimated_variance
                       );
template 
void linear_regression<>(double& constant, double& scale,
		       double& chi_square,
		       double& variance_of_constant,
		       double& variance_of_scale,
		       double& covariance_of_constant_with_scale,
		       const VectorWithOffset<float>& measured_data,
		       const VectorWithOffset<float>& coordinates,
		       const VectorWithOffset<float>& weights,
		       const bool use_estimated_variance
                       );

