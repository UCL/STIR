//
//
/*
  Copyright (C) 2005- 2007, Hammersmith Imanet Ltd
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
  \brief Implementation of stir::inverse_SSRB

  \author Charalampos Tsoumpas
  \author Kris Thielemans
  
*/
#include "stir/ProjData.h"
#include "stir/ProjDataInfo.h"
#include "stir/inverse_SSRB.h"
#include "stir/Sinogram.h"
#include "stir/Bin.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

Succeeded 
inverse_SSRB(ProjData& proj_data_4D,
			 const ProjData& proj_data_3D)
{
	const ProjDataInfo * const proj_data_3D_info_ptr =
		dynamic_cast<ProjDataInfo const * >
		(proj_data_3D.get_proj_data_info_ptr());
	const ProjDataInfo * const proj_data_4D_info_ptr =
		dynamic_cast<ProjDataInfo const * >
		(proj_data_4D.get_proj_data_info_ptr());
	if ((proj_data_3D_info_ptr->get_min_view_num() !=
	     proj_data_4D_info_ptr->get_min_view_num()) ||
	    (proj_data_3D_info_ptr->get_min_view_num() !=
	     proj_data_4D_info_ptr->get_min_view_num()))
	  {
	    warning("inverse_SSRB: incompatible view-information");
	    return Succeeded::no;
	  }
	if ((proj_data_3D_info_ptr->get_min_tangential_pos_num() !=
	     proj_data_4D_info_ptr->get_min_tangential_pos_num()) ||
	    (proj_data_3D_info_ptr->get_min_tangential_pos_num() !=
	     proj_data_4D_info_ptr->get_min_tangential_pos_num()))
	  {
	    warning("inverse_SSRB: incompatible tangential_pos-information");
	    return Succeeded::no;
	  }

 	// keep sinograms out of the loop to avoid reallocations
	// initialise to something because there's no default constructor
	Sinogram<float> sino_4D = 
		proj_data_4D.
		get_empty_sinogram(proj_data_4D.get_min_axial_pos_num(0) , 0);
	Sinogram<float> sino_3D = 
		proj_data_3D.
		get_empty_sinogram(proj_data_3D.get_min_axial_pos_num(0) , 0);
	
	for (int out_segment_num = proj_data_4D.get_min_segment_num(); 
	     out_segment_num <= proj_data_4D.get_max_segment_num();
	     ++out_segment_num)
	  {
		for (int out_ax_pos_num = proj_data_4D.get_min_axial_pos_num(out_segment_num); 
		     out_ax_pos_num  <= proj_data_4D.get_max_axial_pos_num(out_segment_num);
		     ++out_ax_pos_num )
			{		
				sino_4D = proj_data_4D.get_empty_sinogram(out_ax_pos_num, out_segment_num);				
				const float out_m = 
					proj_data_4D_info_ptr->
					get_m(Bin(out_segment_num, 0, out_ax_pos_num, 0));				
				int num_contributing_sinos = 0;
								
				for (int in_ax_pos_num = proj_data_3D.get_min_axial_pos_num(0); 
				     in_ax_pos_num  <= proj_data_3D.get_max_axial_pos_num(0);
				     ++in_ax_pos_num )
				{
					sino_3D = proj_data_3D.get_sinogram(in_ax_pos_num,0);					
					const float in_m = 
						proj_data_3D_info_ptr->get_m(Bin(0, 0, in_ax_pos_num, 0));

					if (fabs(out_m - in_m) < 1E-2)
					{
					        ++num_contributing_sinos;
						sino_4D += sino_3D;	
						if (proj_data_4D.set_sinogram(sino_4D) == Succeeded::no)
							return Succeeded::no;
						break;
					}
					const float in_m_next = in_ax_pos_num == proj_data_3D.get_max_axial_pos_num(0) ? 
						-1000000.F : proj_data_3D_info_ptr->get_m(Bin(0, 0, in_ax_pos_num+1, 0));

					if (fabs(out_m - .5F*(in_m + in_m_next)) < 1E-2)
					{
					        ++num_contributing_sinos;
						sino_4D += sino_3D;
						sino_4D += proj_data_3D.get_sinogram(in_ax_pos_num+1,0);
						sino_4D *= .5F;
						if (proj_data_4D.set_sinogram(sino_4D) == Succeeded::no)
							return Succeeded::no;
						break;
					}
				}
				if (num_contributing_sinos == 0)
				  warning("inverse_SSRB: no sinogram contributes to segment %d, axial_pos_num %d",
					  out_segment_num, out_ax_pos_num);
		}
	}
	return Succeeded::yes;
}
END_NAMESPACE_STIR
