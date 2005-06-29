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
\brief Extension of the 2D sinograms in view direction

  \author Kris Thielemans
  \author Charalampos Tsoumpas
  
	$Date$
	$Revision$
*/
#include "stir/Array.h"
#include "stir/SegmentBySinogram.h"
#include "stir/Sinogram.h"
#include "stir/ProjDataInfo.h"
#include "stir/IndexRange.h"
#include "stir/Bin.h"

START_NAMESPACE_STIR

namespace detail {
inline
Array<2,float>
extend_sinogram(const  Array<2,float>& sino_positive_segment,
				   const  Array<2,float>& sino_negative_segment,
				   const ProjDataInfo& proj_data_info,
				   const int min_view_extension, const int max_view_extension)
{
	//* Check if projdata are from 0 to pi-phi
	bool min_is_extended=false;
	bool max_is_extended=false;
	BasicCoordinate<2,int> min_in, max_in;
	if (!sino_positive_segment.get_regular_range(min_in, max_in))
	{
		warning("input segment 0 should have a regular range");	
	}

	const int org_min_view_num=min_in[1];
	const int org_max_view_num=max_in[1];

	const float min_phi = proj_data_info.get_phi(Bin(0,0,0,0));
	const float max_phi = proj_data_info.get_phi(Bin(0,max_in[1],0,0));

	const float sampling_phi = 
		proj_data_info.get_phi(Bin(0,1,0,0)) - min_phi;
	const int num_views_for_180 = max_in[1]+1; // but max_is_extended?

	if (fabs(min_phi)< .0001)
	{
		min_in[1]-=min_view_extension; 
		min_is_extended=true;			  		
	}
	if (fabs(max_phi-(_PI-sampling_phi))<.001) 
	{		
		max_in[1]+=max_view_extension;
		max_is_extended=true;		
	}


	IndexRange<2> extended_range(min_in, max_in);
	Array<2,float> input_extended_view(extended_range);	  
				
	if (!min_is_extended)
		warning("Minimum view of the original projdata is not 0");
	if (!max_is_extended)
		warning("Maximum view of the original projdata is not 180-phi");

	for (int view_num=min_in[1]; view_num<=max_in[1]; ++view_num)
	{
		bool use_extension=false;
		int symmetric_view_num;
		if (view_num<org_min_view_num && min_is_extended==true)
		{
			use_extension=true;
			symmetric_view_num= view_num + num_views_for_180;
		}
		else if (view_num>org_max_view_num && max_is_extended==true)
		{
			use_extension=true;
			symmetric_view_num = view_num - num_views_for_180;
		}
		if (!use_extension)
			input_extended_view[view_num]=
			sino_positive_segment[view_num]; 
		else
		{
			const int symmetric_min = std::max(min_in[2], -max_in[2]);
			const int symmetric_max = std::min(-min_in[2], max_in[2]);
			for (int tang_num=symmetric_min; tang_num<=symmetric_max; ++tang_num)
				input_extended_view[view_num][tang_num]=
				sino_negative_segment[symmetric_view_num][-tang_num];
			// now do extrapolation where we don't have data
			for (int tang_num=min_in[2]; tang_num<symmetric_min; ++tang_num)
				input_extended_view[view_num][tang_num] =
				input_extended_view[view_num][symmetric_min];
			for (int tang_num=symmetric_max+1; tang_num<=max_in[2]; ++tang_num)
				input_extended_view[view_num][tang_num] =
				input_extended_view[view_num][symmetric_max];
		}		
	} // loop over views
	return input_extended_view;
}
} // end of namespace detail

Array<3,float>
extend_segment(const SegmentBySinogram<float>& sino, 
					const int min_view_extension, const int max_view_extension)
{
	BasicCoordinate<3,int> min, max;
		
		min[1]=sino.get_min_axial_pos_num();
		max[1]=sino.get_max_axial_pos_num();
	    min[2]=sino.get_min_view_num();
		max[2]=sino.get_max_view_num();
		min[3]=sino.get_min_tangential_pos_num();
		max[3]=sino.get_max_tangential_pos_num();
	const IndexRange<3> out_range(min,max);
	Array<3,float> out(out_range);
	for (int ax_pos_num=min[1]; ax_pos_num <=max[1] ; ++ax_pos_num)
	{
		out[ax_pos_num] =
			detail::
			extend_sinogram(sino[ax_pos_num],sino[ax_pos_num], 
			*(sino.get_proj_data_info_ptr()),
			min_view_extension, max_view_extension);
	}
	return out;
}

Array<2,float>
extend_sinogram(const Sinogram<float>& sino,
						const int min_view_extension, const int max_view_extension)
{
	return 
		detail::
	extend_sinogram(sino, sino,
	                         *(sino.get_proj_data_info_ptr()),
							 min_view_extension, max_view_extension);
}

END_NAMESPACE_STIR
