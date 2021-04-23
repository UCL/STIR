//
/*
 Copyright (C) 2009 - 2013, King's College London
 This file is part of STIR.
 
 SPDX-License-Identifier: Apache-2.0
 
 See STIR/LICENSE.txt for details
 */  
/*!
 \file
 \ingroup spatial_transformation
 
 \brief This warps an image.
 \author Charalampos Tsoumpas
*/

#ifndef __stir_warp_image_H__
#define __stir_warp_image_H__

#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/numerics/BSplinesRegularGrid.h"

START_NAMESPACE_STIR

VoxelsOnCartesianGrid<float> 
warp_image(const shared_ptr<DiscretisedDensity<3,float> > & density_sptr, 
           const shared_ptr<DiscretisedDensity<3,float> > & motion_x_sptr, 
           const shared_ptr<DiscretisedDensity<3,float> > & motion_y_sptr, 
           const shared_ptr<DiscretisedDensity<3,float> > & motion_z_sptr, 
           const BSpline::BSplineType spline_type, const bool extend_borders);

END_NAMESPACE_STIR

#endif
