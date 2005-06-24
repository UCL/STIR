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
#include "stir/ProjDataInfoCylindricalNoArcCorr.h" 
#include "stir/Bin.h"
#include "stir/stream.h"
#include "stir/CPUTimer.h"
#include "stir/Sinogram.h"
#include <fstream>
#include <cstdio>
using namespace std;

START_NAMESPACE_STIR

void scale_scatter_per_sinogram(
		ProjData& scaled_scatter_proj_data, 		
		const shared_ptr<ProjData> & scatter_proj_data_sptr, 
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
  			Sinogram<float> scatter_sinogram = scatter_proj_data_sptr->get_sinogram(
				bin.axial_pos_num(),bin.segment_num(),0);		
			Sinogram<float> scaled_sinogram =
				scatter_sinogram;
			 scaled_sinogram*=scale_factors[bin.segment_num()][bin.axial_pos_num()];
		
			 scaled_scatter_proj_data.set_sinogram(scaled_sinogram);
		}
}
	END_NAMESPACE_STIR
		
