//
// $Id$
//
/*!

  \file
  \ingroup buildblock

  \brief Declaration of linear_regression()

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
#ifndef __linear_regression_h__
#define __linear_regression_h__

#include "stir/VectorWithOffset.h"

START_NAMESPACE_STIR

/*!
  \ingroup buildblock
  \brief Implements standard linear regression on VectorWithOffset data

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
inline void 
linear_regression(Value& constant, Value& scale,
		  Value& chi_square,
		  Value& variance_of_constant,
		  Value& variance_of_scale,
		  Value& covariance_of_constant_with_scale,
		  const VectorWithOffset<DataType>& measured_data,
		  const VectorWithOffset<CoordinatesType>& coordinates,
		  const VectorWithOffset<float>& weights,
		  const bool use_estimated_variance = true
		  );

/*!
  \ingroup buildblock
  \brief Implements standard linear regression 

  This function takes the data as iterators for maximum flexibility.
  Note that it is assumed (but not checked) that the
  \a measured_data, \a coordinates and \a weights iterators
  run over the same range.

  \see linear_regression(Value& , Value&,
		  Value& ,
		  Value& ,
		  Value& ,
		  Value& ,
		  const VectorWithOffset<DataType>& ,
		  const VectorWithOffset<CoordinatesType>& ,
		  const VectorWithOffset<float>& ,
		  const bool
		  )
*/

template <class Value, class DataIter, class CoordinatesIter, class WeightsIter>
inline void 
linear_regression(Value& constant, Value& scale,
		  Value& chi_square,
		  Value& variance_of_constant,
		  Value& variance_of_scale,
		  Value& covariance_of_constant_with_scale,
		  DataIter measured_data_begin, DataIter measured_data_end,
		  CoordinatesIter coords_begin, 
		  WeightsIter weights_begin,
		  const bool use_estimated_variance = true);

END_NAMESPACE_STIR

#include "stir/linear_regression.inl"

#endif // __linear_regression_h__


