//
// $Id$
//
/*!

  \file
  \ingroup buildblock

  \brief Implementation of inline functions for linear_regression()

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

START_NAMESPACE_STIR

namespace detail
{
  /* 
     We compute the linear regression using
     Numerical Recipes formulas for numerical stability

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
     (Syy is used to compute chi_square in the current implementation)
 
     The fit is split up into 2 functions:
     - linear_regression_compute_S  
     - linear_regression_compute_fit_from_S   

     The main reason for this is that *compute_S needs to be
     templated in 3 different argument types for maximum
     flexibility. This in practice means it needs to be
     an inline function (to avoid problems at linking time).
     In contrast, linear_regression_compute_fit_from_S   
     is only templated in the Value type, which will be
     float or double anyway.
  */

  // defined in .cxx
template <class Value>
void 
linear_regression_compute_fit_from_S(Value& constant, Value& scale,
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
				     );

template <class DataIter, class CoordinatesIter, class WeightsIter>
inline void 
linear_regression_compute_S(double& S,
			    double& Sx,  
			    double& Sy,
			    double& Syy,
			    double& Stt,
			    double& Sty,
			    const DataIter data_begin, const DataIter data_end,
			    const CoordinatesIter coords_begin, 
			    const WeightsIter weights_begin)
{
  DataIter data_iter = data_begin;
  CoordinatesIter coords_iter = coords_begin;
  WeightsIter weights_iter = weights_begin;
  
  for (; data_iter != data_end; ++data_iter, ++coords_iter, ++weights_iter)
  {
    const double weight = static_cast<double>(*weights_iter);
    S += weight;
    Sx += weight * (*coords_iter);
    Sy += weight * (*data_iter);
    Syy += weight * square(static_cast<double>(*data_iter));
  }

  data_iter = data_begin;
  coords_iter = coords_begin;
  weights_iter = weights_begin;
  
  for (; data_iter != data_end; ++data_iter, ++coords_iter, ++weights_iter)
  {
    const double wti = ( *coords_iter - Sx/S);
    Stt   += *weights_iter * wti * wti;
    Sty += *weights_iter * wti * *data_iter;
  }
}

} // end namespace detail

template <class Value, class DataIter, class CoordinatesIter, class WeightsIter>
inline void 
linear_regression(Value& constant, Value& scale,
		  Value& chi_square,
		  Value& variance_of_constant,
		  Value& variance_of_scale,
		  Value& covariance_of_constant_with_scale,
		  DataIter data_begin, DataIter data_end,
		  CoordinatesIter coords_begin, 
		  WeightsIter weights_begin,
		  const bool use_estimated_variance)
{  
  double Sy = 0;
  double Sx = 0;  
  double S = 0;
  double Syy = 0;
  double Stt = 0;
  double Sty = 0;

  detail::linear_regression_compute_S(S,Sx,Sy, Syy, Stt, Sty,
				      data_begin,  data_end,
				      coords_begin, 
				      weights_begin);

  detail::linear_regression_compute_fit_from_S(constant, scale,
					       chi_square,
					       variance_of_constant,
					       variance_of_scale,
					       covariance_of_constant_with_scale,
					       S,Sx,Sy, Syy, Stt, Sty,
					       data_end - data_begin,
					       use_estimated_variance);
				      
}

template <class Value, class DataType, class CoordinatesType>
inline void 
linear_regression(Value& constant, Value& scale,
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
  

  linear_regression(constant, scale,
		    chi_square,
		    variance_of_constant,
		    variance_of_scale,
		    covariance_of_constant_with_scale,
		    measured_data.begin(), measured_data.end(),
		    coordinates.begin(),
		    weights.begin(),
		    use_estimated_variance);
}

END_NAMESPACE_STIR
