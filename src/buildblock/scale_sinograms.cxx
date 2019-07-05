//
//
/*
  Copyright (C) 2004- 2009, Hammersmith Imanet Ltd
  Copyright (C) 2019, University College London
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
  \brief Implementations of functions defined in scale_sinogram.h

  \author Charalampos Tsoumpas
  \author Kris Thielemans
  
*/	
#include "stir/scale_sinograms.h"
#include "stir/ProjData.h"
#include "stir/ProjDataInfo.h"
#include "stir/Bin.h"
#include "stir/Sinogram.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

Succeeded 
scale_sinograms(
		ProjData& scaled_scatter_proj_data, 		
		const ProjData& scatter_proj_data, 
		const Array<2,float> scale_factors)
{	
  const ProjDataInfo &proj_data_info = 
    dynamic_cast<const ProjDataInfo&> 
    (*scaled_scatter_proj_data.get_proj_data_info_ptr());

  Bin bin;
	
  for (bin.segment_num()=proj_data_info.get_min_segment_num();
       bin.segment_num()<=proj_data_info.get_max_segment_num();
       ++bin.segment_num())				
    for (bin.axial_pos_num()=
	   proj_data_info.get_min_axial_pos_num(bin.segment_num());
	 bin.axial_pos_num()<=proj_data_info.get_max_axial_pos_num(bin.segment_num());
	 ++bin.axial_pos_num())		
      {
	Sinogram<float> scatter_sinogram = scatter_proj_data.get_sinogram(
									  bin.axial_pos_num(),bin.segment_num(),0);		
	Sinogram<float> scaled_sinogram =
	  scatter_sinogram;
	scaled_sinogram*=scale_factors[bin.segment_num()][bin.axial_pos_num()];
		
	if (scaled_scatter_proj_data.set_sinogram(scaled_sinogram) == Succeeded::no)
	  return Succeeded::no;
      }
  return Succeeded::yes;
}


Array<2,float>
get_scale_factors_per_sinogram(const ProjData& numerator_proj_data, 
			       const ProjData& denominator_proj_data, 
			       const ProjData& weights_proj_data) 
{
	
  const ProjDataInfo &proj_data_info = 
    dynamic_cast<const ProjDataInfo&> 
    (*weights_proj_data.get_proj_data_info_ptr());

  Bin bin;

  // scale factor to use when the denominator is zero
  const float default_scale = 1.F;

  IndexRange2D sinogram_range(proj_data_info.get_min_segment_num(),proj_data_info.get_max_segment_num(),0,0);
  for (int segment_num=proj_data_info.get_min_segment_num();
       segment_num<=proj_data_info.get_max_segment_num();
       ++segment_num)
    {
      sinogram_range[segment_num].resize(
					 proj_data_info.get_min_axial_pos_num(segment_num),
					 proj_data_info.get_max_axial_pos_num(segment_num) );
    }
  Array<2,float> total_in_denominator(sinogram_range),
    total_in_numerator(sinogram_range);
  Array<2,float> scale_factors(sinogram_range);
  for (bin.segment_num()=proj_data_info.get_min_segment_num();
       bin.segment_num()<=proj_data_info.get_max_segment_num();
       ++bin.segment_num())		
    for (bin.axial_pos_num()=
	   proj_data_info.get_min_axial_pos_num(bin.segment_num());
	 bin.axial_pos_num()<=proj_data_info.get_max_axial_pos_num(bin.segment_num());
	 ++bin.axial_pos_num())
      {
	const Sinogram<float> weights = 
	  weights_proj_data.get_sinogram(bin.axial_pos_num(),bin.segment_num());
	const Sinogram<float> denominator_sinogram = 
	  denominator_proj_data.get_sinogram(bin.axial_pos_num(),bin.segment_num());
	const Array<2,float> weighted_denominator_sinogram =  denominator_sinogram * weights;
	const Array<2,float> weighted_numerator_sinogram = 
	  numerator_proj_data.get_sinogram(bin.axial_pos_num(),bin.segment_num()) * weights;
	total_in_denominator[bin.segment_num()][bin.axial_pos_num()] = weighted_denominator_sinogram.sum();
	total_in_numerator[bin.segment_num()][bin.axial_pos_num()] =  weighted_numerator_sinogram.sum();
      
	if (denominator_sinogram.sum()==0)
	  {  
	    scale_factors[bin.segment_num()][bin.axial_pos_num()] = default_scale;
	  }
	else
	  {
	    if (total_in_denominator[bin.segment_num()][bin.axial_pos_num()]<=
		denominator_sinogram.sum()/
		(proj_data_info.get_num_views() * proj_data_info.get_num_tangential_poss()) * .001
		)
	      {
		warning("Problem at segment %d, axial pos %d in finding sinogram scaling factor.\n"
                        "Weighted data in denominator %g is very small compared to total in sinogram %g.\n"
                        "Adjust weights?.\n"
                        "I will use scale factor %g",
                        bin.segment_num(),bin.axial_pos_num(),
                        total_in_denominator[bin.segment_num()][bin.axial_pos_num()],
                        denominator_sinogram.sum(),
                        default_scale
		      );
                scale_factors[bin.segment_num()][bin.axial_pos_num()] = default_scale;
	      }
            else
              {
                scale_factors[bin.segment_num()][bin.axial_pos_num()] =
                  total_in_numerator[bin.segment_num()][bin.axial_pos_num()]/
                  total_in_denominator[bin.segment_num()][bin.axial_pos_num()];
              }
	  }
      }
  
  return scale_factors;		
}

END_NAMESPACE_STIR
		
