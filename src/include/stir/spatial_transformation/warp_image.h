//
/*
 Copyright (C) 2009 - 2013, King's College London
 This file is part of STIR.
 
 This file is free software; you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation; either version 2.3 of the License, or
 (at your option) any later version.
 
 This file is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Lesser General Public License for more details.
 
 See STIR/LICENSE.txt for details
 */  
/*!
 \file
 \ingroup spatial_transformation
 
 \brief This warps an image.
 \author Charalampos Tsoumpas
 $Date$
 $Revision$
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
