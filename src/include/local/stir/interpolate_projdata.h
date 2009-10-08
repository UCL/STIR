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
  \author Kris Thielemans

  $Date$
  $Revision$
*/

#include "stir/numerics/BSplines.h"
//#include "stir/numerics/BSplinesRegularGrid.h"

START_NAMESPACE_STIR

class ProjData;
template <int num_dimensions, class T> class BasicCoordinate;
template <class elemT> class Sinogram;
template <class elemT> class SegmentBySinogram;

shared_ptr<ProjDataInfo>
make_non_interleaved_proj_data_info(const ProjDataInfo& proj_data_info);

void
make_non_interleaved_sinogram(Sinogram<float>& out_sinogram,
			      const Sinogram<float>& in_sinogram);

Sinogram<float>
make_non_interleaved_sinogram(const ProjDataInfo& non_interleaved_proj_data_info,
			      const Sinogram<float>& in_sinogram);

void
make_non_interleaved_segment(SegmentBySinogram<float>& out_segment,
			     const SegmentBySinogram<float>& in_segment);

SegmentBySinogram<float>
make_non_interleaved_segment(const ProjDataInfo& non_interleaved_proj_data_info,
			     const SegmentBySinogram<float>& in_segment);

//! \brief Perform B-Splines Interpolation
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
//@{
/*!						  
\ingroup projdata
\brief Extension of the 2D sinograms in view direction.
	Functions that interpolate the given input projection 3D data to the given output projection
	3D template using B-Splines interpolators.
*/
Succeeded 
interpolate_projdata(ProjData& proj_data_out,
		     const ProjData& proj_data_in, 
		     const BSpline::BSplineType this_type,
		     const bool remove_interleaving = false);
Succeeded 
interpolate_projdata(ProjData& proj_data_out,
		     const ProjData& proj_data_in,
		     const BasicCoordinate<3, BSpline::BSplineType> & this_type,
		     const bool remove_interleaving = false);
//@}

END_NAMESPACE_STIR



