//
//
/*!
  \file 
  \ingroup evaluation

  \brief Declaration of various function that computes ROI values 

  \author Kris Thielemans
*/
/*
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
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
#ifndef __stir_evaluation_compute_ROI_values__H__
#define __stir_evaluation_compute_ROI_values__H__

#include "stir/evaluation/ROIValues.h"

START_NAMESPACE_STIR

template <typename coordT> class CartesianCoordinate2D;
template <typename coordT> class CartesianCoordinate3D;
template <typename elemT> class VectorWithOffset;
template <int num_dimensions, typename elemT> class DiscretisedDensity;
class Shape3D;

/*! \ingroup evaluation
    \name Functions to compute ROI values
    
    Shapes are first discretised using Shape3D::construct_volume. This can make fuzzy 
    boundaries (when the \a num_samples argument is not (1,1,1), or when DiscretisedShape3D 
    needs zooming). Mean and stddev are computed using weighted versions, taking this smoothness 
    into account, while ROI_min and max are ignore those weights.
*/
//@{

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

// end of doxygen group
//@}

END_NAMESPACE_STIR

#endif
