//
// $Id$
//
/*!
  \file Declaration of various function that computes ROI values
  \ingroup eval_buildblock

  \brief 

  \author Kris Thielemans
  $Date$
  $Revision$
*/
#ifndef __tomo_eval_buildblock_compute_ROI_values__H__
#define __tomo_eval_buildblock_compute_ROI_values__H__

#include "local/tomo/eval_buildblock/ROIValues.h"

START_NAMESPACE_TOMO

template <typename coordT> class CartesianCoordinate2D;
template <typename coordT> class CartesianCoordinate3D;
template <typename elemT> class VectorWithOffset;
template <int num_dimensions, typename elemT> class DiscretisedDensity;
class Shape3D;

// TODO doc

void
compute_ROI_values_per_plane(VectorWithOffset<ROIValues>& values, 
			     const DiscretisedDensity<3,float>& image, 
                             const Shape3D& shape,
                             const CartesianCoordinate3D<int>& num_samples);
ROIValues
compute_total_ROI_values(const VectorWithOffset<ROIValues>& values);

ROIValues
compute_total_ROI_values(const DiscretisedDensity<3,float>& image,
                         const Shape3D& shape, 
                         const CartesianCoordinate3D<int>& num_samples
			 );

// function that calculate the 
void
compute_plane_range_ROI_values_per_plane(VectorWithOffset<ROIValues>& values, 
			     const DiscretisedDensity<3,float>& image,
			     const CartesianCoordinate2D<int>& plane_range,
                             const Shape3D& shape,
                             const CartesianCoordinate3D<int>& num_samples);

float
compute_CR_hot(ROIValues& val1, ROIValues& val2);
float
compute_CR_cold(ROIValues& val1, ROIValues& val2);
float
compute_uniformity(ROIValues& val);

VectorWithOffset<float>
compute_CR_hot_per_plane(VectorWithOffset<ROIValues>& val1,VectorWithOffset<ROIValues>& val2);

VectorWithOffset<float>
compute_CR_cold_per_plane(VectorWithOffset<ROIValues>& val1,VectorWithOffset<ROIValues>& val2);

VectorWithOffset<float>
compute_uniformity_per_plane(VectorWithOffset<ROIValues>& val);


END_NAMESPACE_TOMO

#endif
