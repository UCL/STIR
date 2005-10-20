//
// $Id$
//
/*!
\file
\ingroup scatter
\brief Implementations of functions defined in Scatter.h

\author Charalampos Tsoumpas
\author Kris Thielemans
  
$Date$
$Revision$
	
Copyright (C) 2004- $Date$, Hammersmith Imanet
See STIR/LICENSE.txt for details
*/
#include "local/stir/Scatter.h"
#include "stir/ProjData.h"
#include "stir/ProjDataInfo.h"
#include "stir/Bin.h"
#include "stir/Viewgram.h"
#include <stir/IndexRange2D.h> 
#include <iostream>

START_NAMESPACE_STIR

Array<2,float>
scale_factors_per_viewgram(const ProjData& emission_proj_data, 
			   const ProjData & scatter_proj_data, 
			   const ProjData& att_proj_data, 
			   const float attenuation_threshold,
			   const float mask_radius_in_mm
			   ) 
{
	
  const ProjDataInfo &proj_data_info = 
    dynamic_cast<const ProjDataInfo&> 
    (*att_proj_data.get_proj_data_info_ptr());

  Bin bin;
	
  IndexRange2D viewgram_range(proj_data_info.get_min_segment_num(),proj_data_info.get_max_segment_num(),0,0);
  for (int segment_num=proj_data_info.get_min_segment_num();
       segment_num<=proj_data_info.get_max_segment_num();
       ++segment_num)
    {
      viewgram_range[segment_num].resize(
					 proj_data_info.get_min_view_num(),
					 proj_data_info.get_max_view_num() );
    }
  Array<2,float> total_outside_scatter(viewgram_range),
    total_outside_emission(viewgram_range);
  Array<2,float> scale_factors(viewgram_range);
  for (bin.segment_num()=proj_data_info.get_min_segment_num();
       bin.segment_num()<=proj_data_info.get_max_segment_num();
       ++bin.segment_num())		
    for (bin.view_num()=
	   proj_data_info.get_min_view_num();
	 bin.view_num()<=proj_data_info.get_max_view_num();
	 ++bin.view_num())
      {
	const Viewgram<float> scatter_viewgram = scatter_proj_data.get_viewgram(
										bin.view_num(),bin.segment_num());
	const Viewgram<float> emission_viewgram = emission_proj_data.get_viewgram(
										  bin.view_num(),bin.segment_num());
	const Viewgram<float> att_viewgram = att_proj_data.get_viewgram(
									bin.view_num(),bin.segment_num());
	int count=0;
	for (bin.axial_pos_num()=proj_data_info.get_min_axial_pos_num(bin.segment_num());
	     bin.axial_pos_num()<=proj_data_info.get_max_axial_pos_num(bin.segment_num());
	     ++bin.axial_pos_num())
	  for (bin.tangential_pos_num()=
		 proj_data_info.get_min_tangential_pos_num();
	       bin.tangential_pos_num()<=
		 proj_data_info.get_max_tangential_pos_num();
	       ++bin.tangential_pos_num())
	    if (att_viewgram[bin.axial_pos_num()][bin.tangential_pos_num()]<attenuation_threshold &&
		(mask_radius_in_mm<0 || mask_radius_in_mm>= std::fabs(scatter_proj_data.get_proj_data_info_ptr()->get_s(bin))))
	      {						
		++count;
		total_outside_scatter[bin.segment_num()][bin.view_num()] += 
		  scatter_viewgram[bin.axial_pos_num()][bin.tangential_pos_num()] ;					
		total_outside_emission[bin.segment_num()][bin.view_num()] += 
		  emission_viewgram[bin.axial_pos_num()][bin.tangential_pos_num()] ;										
	      }
#ifndef NDEBUG
	std::cout << total_outside_emission[bin.segment_num()][bin.view_num()] << " " <<
	  total_outside_scatter[bin.segment_num()][bin.view_num()] << '\n';
#endif
	std::cout << count << " bins in mask\n";

	if (total_outside_scatter[bin.segment_num()][bin.view_num()]<=
	    scatter_viewgram.sum()/
	    (proj_data_info.get_num_axial_poss(bin.segment_num()) * proj_data_info.get_num_tangential_poss()) * .001
	    )
	  {
	    error("Problem in finding scatter scaling factor.\n"
		  "Scatter data in mask %g too small compared to total in viewgram %g.\n"
		  "Adjust threshold?",
		  total_outside_scatter[bin.segment_num()][bin.view_num()],
		  scatter_viewgram.sum()
		  );
	  }
	scale_factors[bin.segment_num()][bin.view_num()] = 
	  total_outside_emission[bin.segment_num()][bin.view_num()]/
	  total_outside_scatter[bin.segment_num()][bin.view_num()];
      }
			
  return scale_factors;		
}
END_NAMESPACE_STIR
		
