//
// $Id$
//
/*!
  \file
  \ingroup scatter
  \brief Implementations of functions defined in Scatter.h

  \author Charalampos Tsoumpas
  \author Pablo Aguiar
  \author Kris Thielemans
  
  $Date$
  $Revision$
	
  Copyright (C) 2004- $Date$, Hammersmith Imanet
  See STIR/LICENSE.txt for details
*/
#include "local/stir/Scatter.h"
#include "stir/ProjData.h"
#include "stir/ProjDataInfo.h"
#include "stir/Succeeded.h"
#include "stir/Bin.h"
#include "stir/Viewgram.h"

START_NAMESPACE_STIR

void scale_scatter_per_viewgram(
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
    for (bin.view_num()=
	   proj_data_info.get_min_view_num();
	 bin.view_num()<=proj_data_info.get_max_view_num();
	 ++bin.view_num())		
      {
	Viewgram<float> scatter_viewgram = scatter_proj_data.get_viewgram(
									  bin.view_num(),bin.segment_num(),0);		
	Viewgram<float> scaled_viewgram =
	  scatter_viewgram;
	scaled_viewgram*=scale_factors[bin.segment_num()][bin.view_num()];
		
	scaled_scatter_proj_data.set_viewgram(scaled_viewgram);
      }
}
END_NAMESPACE_STIR
		
