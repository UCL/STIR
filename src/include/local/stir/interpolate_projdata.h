//
// $Id$
//
/*
    Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
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
/*
  \ingroup projdata
  \file Declaration of stir::interpolate_projdata
  
  \author Charalampos Tsoumpas

  $Date$
  $Revision$
*/


#include "local/stir/BSplines.h"
#include "local/stir/BSplinesRegularGrid.h"


START_NAMESPACE_STIR

class ProjData;
class Succeeded;
//! Perform B-Splines Interpolation
/*! 
  \ingroup projdata
  \param[out] proj_data_out Its projection_data_info is used to 
  determine output characteristics. Data will be 'put' in here using 
  ProjData::set_sinogram().
  \param[in] proj_data_in input data

  The STIR implementation of interpolating 3D (for the moment) projdata is a generalisation that applies 
  B-Splines Interpolation to projdata supposing that every dimension is a regular grid. For instance, for 
  a 3D dataset, interpolating can produce a new expanded 3D dataset based on the given information 
  (proj_data_out). This mostly is useful in the scatter sinogram expansion.

  See STIR documentation about B-Spline interpolation or scatter correction.     
*/  
Succeeded 
interpolate_projdata(ProjData& proj_data_out,
		const ProjData& proj_data_in,
		const BasicCoordinate<3, BSpline::BSplineType> & this_type);
/*
Succeeded 
interpolate_projdata(ProjData& proj_data_out,
		const ProjData& proj_data_in, const BSplineType this_type);*/


END_NAMESPACE_STIR

