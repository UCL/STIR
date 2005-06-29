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
/*!
\file
\ingroup projdata
//! Perform B-Splines Interpolation

  \author Charalampos Tsoumpas
  \author Kris Thielemans
  
	$Date$
	$Revision$
*/
#include "stir/ProjData.h"
#include "stir/ProjDataInfo.h"
#include "stir/IndexRange.h"
#include "stir/BasicCoordinate.h"
#include "stir/Sinogram.h"
#include "stir/SegmentBySinogram.h"
#include "stir/Succeeded.h"
#include "local/stir/BSplines.h"
#include "local/stir/BSplinesRegularGrid.h"
#include "local/stir/interpolate_projdata.h"
#include "local/stir/extend_projdata.h"
#include "local/stir/sample_array.h"

START_NAMESPACE_STIR
using namespace BSpline;

Succeeded 
interpolate_projdata(ProjData& proj_data_out,
const ProjData& proj_data_in, const BSplineType these_types)
{
BasicCoordinate<3, BSplineType> these_types_3; 
these_types_3[1]=these_types_3[2]=these_types_3[3]=these_types;
interpolate_projdata(proj_data_out,proj_data_in,these_types_3);
return Succeeded::yes;
}

Succeeded 
interpolate_projdata(ProjData& proj_data_out,
					 const ProjData& proj_data_in,
					 const BasicCoordinate<3, BSplineType> & these_types)
{
	const ProjDataInfo & proj_data_in_info =
		*proj_data_in.get_proj_data_info_ptr();
	const ProjDataInfo & proj_data_out_info =
		*proj_data_out.get_proj_data_info_ptr();
	
    BSpline::BSplinesRegularGrid<3, float, float> proj_data_interpolator(these_types);
	BasicCoordinate<3, double>  offset,  step  ;
	
	// find relation between out_index and in_index such that they correspond to the same physical position
	// out_index * m_zoom + m_offset = in_index
    const float in_sampling_m = proj_data_in_info.get_sampling_in_m(Bin(0,0,0,0));
	const float out_sampling_m = proj_data_out_info.get_sampling_in_m(Bin(0,0,0,0));
	// offset in 'in' index units
    offset[1] = 
		(proj_data_in_info.get_m(Bin(0,0,0,0)) -
		proj_data_out_info.get_m(Bin(0,0,0,0))) / in_sampling_m;
	step[1]=
		out_sampling_m/in_sampling_m;
		
	const float in_sampling_phi = 
		proj_data_in_info.get_phi(Bin(0,1,0,0)) - proj_data_in_info.get_phi(Bin(0,0,0,0));
	const float out_sampling_phi = 
		proj_data_out_info.get_phi(Bin(0,1,0,0)) - proj_data_out_info.get_phi(Bin(0,0,0,0));
	
	offset[2] = 
		(proj_data_in_info.get_phi(Bin(0,0,0,0)) - proj_data_out_info.get_phi(Bin(0,0,0,0))) / in_sampling_phi;
	step[2] =
		out_sampling_phi/in_sampling_phi;
	
	const float in_sampling_s = proj_data_in_info.get_sampling_in_s(Bin(0,0,0,0));// It was Bin instead of bin
	const float out_sampling_s = proj_data_out_info.get_sampling_in_s(Bin(0,0,0,0));
	offset[3] = 
		(proj_data_out_info.get_s(Bin(0,0,0,0)) -
		proj_data_in_info.get_s(Bin(0,0,0,0))) / in_sampling_s;
	step[3]=
		out_sampling_s/in_sampling_s;
	
	// initialise interpolator
	{
        const Array<3,float> input_extended_view = extend_segment(
			proj_data_in.get_segment_by_sinogram(0), 2, 2);
		proj_data_interpolator.set_coef(input_extended_view);
	}
	
	// now do interpolation		
 	SegmentBySinogram<float> sino_3D_out = proj_data_out.get_empty_segment_by_sinogram(0) ;
   sample_at_regular_array(sino_3D_out, proj_data_interpolator, offset, step);

	proj_data_out.set_segment(sino_3D_out);
	if (proj_data_out.set_segment(sino_3D_out) == Succeeded::no)
						  return Succeeded::no;	   
	return Succeeded::yes;
}

END_NAMESPACE_STIR
