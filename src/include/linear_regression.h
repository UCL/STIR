//
// $Id$: $Date$
//
/*!

  \file
  \ingroup buildblock

  \brief Declaration of linear_regression()

  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/
#ifndef __linear_regression_h__
#define __linear_regression_h__

#include "VectorWithOffset.h"

START_NAMESPACE_TOMO

/*!
  \ingroup buildblock
  \brief Implements standard linear regression

   The linear_regression function does straightforward 
   (1 dimensional) weighted least squares fitting.

   \par input  
   
    3 VectorWithOffsets of measured_data, coordinates, weights<br>
    1 optional boolean value use_estimated_variance (default=true)

   \par output
   
     fitted parameters : constant, scale<br>
     goodness of fit measures : chi_square and the (co)variances
   
   This solves the minimisation problem:
\verbatim
     Find constant, scale such that
       chi_square = 
          sum_i weights[i]*
                (constant + scale*coordinates[i] - measured_data[i])^2
     is minimal.
\endverbatim

   When use_estimated_variance == false, the (co)variances are computed 
   assuming that 1/weights[i] is the standard deviation on measured_data[i]. 
   In particular, this means that the (co)variances depend only on the 
   weights and the coordinates.

   Alternatively, when use_estimated_variance == true, the weights are 
   considered to be really only proportional to the 
   1/variance. Then the estimated variance is used to get sensible
   estimates of the errors:
   \verbatim
     estimated_variance = chi_square/(measured_data.get_length() - 2)
     estimated_covariance_matrix = original_covariance_matrix * estimated_variance
   \endverbatim
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

END_NAMESPACE_TOMO

#endif // __linear_regression_h__


