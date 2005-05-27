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
\brief Implementation of inverse_SSRB

  \author Charalmapos Tsoumpas
  \author Kris Thielemans
  
	$Date$
	$Revision$
*/
#include "stir/ProjDataFromStream.h"
#include "stir/ProjDataInterfile.h"
#include "stir/ProjDataInfoCylindrical.h"
#include "local/stir/inverse_SSRB.h"
#include "stir/Sinogram.h"
#include "stir/Bin.h"
#include "stir/round.h"
#include <fstream>
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::fstream;
using std::min;
using std::max;
#endif

START_NAMESPACE_STIR

void 
inverse_SSRB(ProjData& proj_data_3D,
			 const ProjData& proj_data_2D)
{
	const ProjDataInfoCylindrical * const proj_data_2D_info_ptr =
		dynamic_cast<ProjDataInfoCylindrical const * >
		(proj_data_2D.get_proj_data_info_ptr());
	if (proj_data_2D_info_ptr== NULL)
	{
		error("inverse_SSRB works only on segments with proj_data_info of "
			"type ProjDataInfoCylindrical\n");
	}
	const ProjDataInfoCylindrical * const proj_data_3D_info_ptr =
		dynamic_cast<ProjDataInfoCylindrical const * >
		(proj_data_3D.get_proj_data_info_ptr());
	if (proj_data_3D_info_ptr== NULL)
	{
		error("inverse_SSRB works only on segments with proj_data_info of "
			"type ProjDataInfoCylindrical\n");
	}
	/*
	const int num_views_to_combine =
    proj_data_2D.get_num_views()/ proj_data_3D.get_num_views();
	*/
	
	if (proj_data_2D.get_min_view_num()!=0 || proj_data_3D.get_min_view_num()!=0)
		error ("inverse_SSRB can only mash views when min_view_num==0\n");
	if (proj_data_2D.get_num_views() % proj_data_3D.get_num_views())
		error ("inverse_SSRB can only mash views when out_num_views divides in_num_views\n");
	
	for (int out_segment_num = proj_data_3D.get_min_segment_num(); 
	out_segment_num <= proj_data_3D.get_max_segment_num();
	++out_segment_num)
    {
		// find range of input segments that fit in the current output segment
		int in_min_segment_num = proj_data_2D.get_max_segment_num();
		int in_max_segment_num = proj_data_2D.get_min_segment_num();
		{
			// this the only place where we need ProjDataInfoCylindrical
			// Presumably for other types, there'd be something equivalent (say range of theta)
			const int out_min_ring_diff = 
				proj_data_3D_info_ptr->get_min_ring_difference(out_segment_num);
			const int out_max_ring_diff = 
				proj_data_3D_info_ptr->get_max_ring_difference(out_segment_num);
			for (int in_segment_num = proj_data_2D.get_min_segment_num(); 
			in_segment_num <= proj_data_2D.get_max_segment_num();
			++in_segment_num)
			{
				const int in_min_ring_diff = 
					proj_data_2D_info_ptr->get_min_ring_difference(in_segment_num);
				const int in_max_ring_diff = 
					proj_data_2D_info_ptr->get_max_ring_difference(in_segment_num);
				if (in_min_ring_diff >= out_min_ring_diff &&
					in_max_ring_diff <= out_max_ring_diff)
				{
					// it's a in_segment that should be rebinned in the out_segment
					if (in_min_segment_num > in_segment_num)		  
						in_min_segment_num = in_segment_num;
					if (in_max_segment_num < in_segment_num)
						in_max_segment_num = in_segment_num;
				}
				else if (in_min_ring_diff > out_max_ring_diff || in_max_ring_diff < out_min_ring_diff)
				{
					// this one is outside the range of the out_segment
				}
				else
				{
					error("inverse_SSRB called with in and out ring difference ranges that overlap:\n"
						"in_segment %d has ring diffs (%d,%d)\n"
						"out_segment %d has ring diffs (%d,%d)\n",
						in_segment_num, in_min_ring_diff, in_max_ring_diff,
						in_segment_num, out_min_ring_diff, out_max_ring_diff);
				}
			}
			
			// keep sinograms out of the loop to avoid reallocations
			// initialise to something because there's no default constructor
			
			Sinogram<float> sino_3D = 
				proj_data_3D.get_empty_sinogram(
				proj_data_3D.get_min_axial_pos_num(out_segment_num) , out_segment_num);
			Sinogram<float> sino_2D = 
				proj_data_2D.get_sinogram(
			proj_data_2D.get_min_axial_pos_num(out_segment_num) , 0);
			
			for (int out_ax_pos_num = proj_data_3D.get_min_axial_pos_num(out_segment_num); 
			out_ax_pos_num  <= proj_data_3D.get_max_axial_pos_num(out_segment_num);
			++out_ax_pos_num )
			{
				sino_3D= proj_data_3D.get_empty_sinogram(out_ax_pos_num, out_segment_num);
				
				const float out_m = proj_data_3D_info_ptr->get_m(
					Bin(out_segment_num, 0, out_ax_pos_num, 0));				
				
				
				for (int in_ax_pos_num = proj_data_3D.get_min_axial_pos_num(0); 
				in_ax_pos_num  <= proj_data_3D.get_max_axial_pos_num(0);
				++in_ax_pos_num )
				{
					const float in_m = proj_data_2D_info_ptr->get_m(
						Bin(0, 0, in_ax_pos_num, 0));
					if (fabs(out_m - in_m) < 1E-4)
					{					
						if (out_segment_num>=0)
							sino_3D += proj_data_2D.get_sinogram(in_ax_pos_num, 0);						
						else
						{
							sino_2D += proj_data_2D.get_sinogram(in_ax_pos_num, 0);
							for (int in_view_num=proj_data_2D.get_min_view_num();
							in_view_num <= proj_data_2D.get_max_view_num();
							++in_view_num)
								for (int tangential_pos_num=
									proj_data_3D.get_min_tangential_pos_num();
								tangential_pos_num <=  
									proj_data_3D.get_max_tangential_pos_num();
								++tangential_pos_num)							
								{							
									sino_3D[in_view_num][tangential_pos_num] = 
										sino_2D[in_view_num][-tangential_pos_num];									
								}						
						}
					}
					continue; 
				}
				/*
				if (num_in_ax_pos==0)
					warning("inverse_SSRB: no sinograms contributing to output segment %d, ax_pos %d\n",
					out_segment_num, out_ax_pos_num);*/
				proj_data_3D.set_sinogram(sino_3D);
			}
		}
	}
}
END_NAMESPACE_STIR
