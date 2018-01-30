//
//
/*
    Copyright (C) 2005- 2009, Hammersmith Imanet Ltd
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
  \author Kris Thielemans

*/

#include "stir/numerics/BSplines.h"

START_NAMESPACE_STIR

class ProjData;
template <int num_dimensions, class T> class BasicCoordinate;
template <class elemT> class Sinogram;
template <class elemT> class SegmentBySinogram;


//! \brief Perform B-Splines Interpolation
/*! 
  \ingroup projdata
  \param[out] proj_data_out Its projection_data_info is used to 
  determine output characteristics. Data will be 'put' in here using 
  ProjData::set_sinogram().
  \param[in] proj_data_in input data 
  \param[in] spline_type determines which type of BSpline will be used
  \param[in] remove_interleaving 
  The STIR implementation of interpolating 3D (for the moment) projdata is a generalisation that applies 
  B-Splines Interpolation to projdata supposing that every dimension is a regular grid. For instance, for 
  a 3D dataset, interpolating can produce a new expanded 3D dataset based on the given information 
  (proj_data_out). This mostly is useful in the scatter sinogram expansion.

  See STIR documentation about B-Spline interpolation or scatter correction.     

  \todo This currently only works for direct sinograms (i.e. segment 0).
  \warning Because of the boundary conditions in the B-spline interpolation,
  strange results can occur if the output sinogram has a larger range than 
  the input sinogram.
*/  
//@{
Succeeded 
interpolate_projdata(ProjData& proj_data_out,
		     const ProjData& proj_data_in, 
		     const BSpline::BSplineType spline_type,
		     const bool remove_interleaving = false,
		     const bool use_view_offset = false);
Succeeded 
interpolate_projdata(ProjData& proj_data_out,
		     const ProjData& proj_data_in,
		     const BasicCoordinate<3, BSpline::BSplineType> & spline_type,
		     const bool remove_interleaving = false,
		     const bool use_view_offset = false);
//@}

END_NAMESPACE_STIR



