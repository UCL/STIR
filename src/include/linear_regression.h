//
// $Id$: $Date$
//

#ifndef __linear_regression_h__
#define __linear_regression_h__

/*
  Linear regression function
  
  Kris Thielemans, 08/12/1999
*/

#include "VectorWithOffset.h"

/*
   The linear_regression function does straightforward 
   (1 dimensional) weighted least squares fitting.
   input  : 3 VectorWithOffsets of measured_data, coordinates, weights
            1 optional boolean value use_estimated_variance (default=true)
   output : fitted parameters : constant, scale
            goodness of fit measures : chi_square and the (co)variances
   
   This solves the minimisation problem:
     Find constant, scale such that
       chi_square = 
          sum_i weights[i]*
                (constant + scale*coordinates[i] - measured_data[i])^2
     is minimal.

   When use_estimated_variance == false, the (co)variances are computed 
   assuming that 1/weights[i] is the standard deviation on measured_data[i]. 
   In particular, this means that the (co)variances depend only on the 
   weights and the coordinates.

   Alternatively, when use_estimated_variance == true, the weights are 
   considered to be really only proportional to the 
   1/variance. Then the estimated variance is used to get sensible
   estimates of the errors:
     estimated_variance = chi_square/(measured_data.get_length() - 2)
     estimated_covariance_matrix = original_covariance_matrix * estimated_variance
*/

template <class Value, class DataType, class CoordinatesType>
void linear_regression(Value& constant, Value& scale,
		       Value& chi_square,
		       Value& variance_of_constant,
		       Value& variance_of_scale,
		       Value& covariance_of_constant_with_scale,
		       const VectorWithOffset<DataType>& measured_data,
		       const VectorWithOffset<CoordinatesType>& coordinates,
		       const VectorWithOffset<float>& weights,
		       const bool use_estimated_variance = true
                       );
#endif // __linear_regression_h__

