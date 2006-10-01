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
#include "stir/Bin.h"
#include "stir/CPUTimer.h"
#include "stir/Sinogram.h"
#include "stir/IndexRange2D.h"
#include "stir/ArrayFilter1DUsingConvolution.h"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <boost/lambda/lambda.hpp>

using boost::lambda::_1;

START_NAMESPACE_STIR

Array<2,float>
scale_factors_per_sinogram(const ProjData& emission_proj_data, 
			   const ProjData & scatter_proj_data, 
			   const ProjData& att_proj_data, 
			   const float attenuation_threshold,
#ifdef SCFOLD
			   const float mask_radius_in_mm
#else
			   const std::size_t back_off
#endif
			   ) 
{
	
  const ProjDataInfo &proj_data_info = 
    dynamic_cast<const ProjDataInfo&> 
    (*att_proj_data.get_proj_data_info_ptr());

  Bin bin;
	
  IndexRange2D sinogram_range(proj_data_info.get_min_segment_num(),proj_data_info.get_max_segment_num(),0,0);
  for (int segment_num=proj_data_info.get_min_segment_num();
       segment_num<=proj_data_info.get_max_segment_num();
       ++segment_num)
    {
      sinogram_range[segment_num].resize(
					 proj_data_info.get_min_axial_pos_num(segment_num),
					 proj_data_info.get_max_axial_pos_num(segment_num) );
    }
  Array<2,float> total_outside_scatter(sinogram_range),
    total_outside_emission(sinogram_range);
  Array<2,float> scale_factors(sinogram_range);
  for (bin.segment_num()=proj_data_info.get_min_segment_num();
       bin.segment_num()<=proj_data_info.get_max_segment_num();
       ++bin.segment_num())		
    for (bin.axial_pos_num()=
	   proj_data_info.get_min_axial_pos_num(bin.segment_num());
	 bin.axial_pos_num()<=proj_data_info.get_max_axial_pos_num(bin.segment_num());
	 ++bin.axial_pos_num())
      {
	const Sinogram<float> scatter_sinogram = scatter_proj_data.get_sinogram(
										bin.axial_pos_num(),bin.segment_num());
	const Sinogram<float> emission_sinogram = emission_proj_data.get_sinogram(
										  bin.axial_pos_num(),bin.segment_num());
	const Sinogram<float> att_sinogram = att_proj_data.get_sinogram(
									bin.axial_pos_num(),bin.segment_num());

	int  count=0;
	for (bin.view_num()=proj_data_info.get_min_view_num();
	     bin.view_num()<=proj_data_info.get_max_view_num();
	     ++bin.view_num())
	  {
#ifdef SCFOLD
	  for (bin.tangential_pos_num()=
		 proj_data_info.get_min_tangential_pos_num();
	       bin.tangential_pos_num()<=
		 proj_data_info.get_max_tangential_pos_num();
	       ++bin.tangential_pos_num())
	    if (att_sinogram[bin.view_num()][bin.tangential_pos_num()]<attenuation_threshold &&
		(mask_radius_in_mm<0 || mask_radius_in_mm>= std::fabs(scatter_proj_data.get_proj_data_info_ptr()->get_s(bin))))
	      {
		++count;
		total_outside_scatter[bin.segment_num()][bin.axial_pos_num()] += 
		  scatter_sinogram[bin.view_num()][bin.tangential_pos_num()] ;					
		total_outside_emission[bin.segment_num()][bin.axial_pos_num()] += 
		  emission_sinogram[bin.view_num()][bin.tangential_pos_num()] ;										
	      }
#else
	  const Array<1,float>& att_line=att_sinogram[bin.view_num()];
	  std::size_t mask_left_size =
	    std::max(std::size_t(0),
		     static_cast<std::size_t>
		     (std::find_if(att_line.begin(), att_line.end(),
				   _1>=attenuation_threshold) -
		      att_line.begin()) -
		     back_off);
	  std::size_t mask_right_size =
	    std::max(std::size_t(0),
		     static_cast<std::size_t>
		     (std::find_if(att_line.rbegin(), att_line.rend() - mask_left_size,
				   _1>=attenuation_threshold) -
		      att_line.rbegin()) -
		     back_off);
	  std::cout << "mask sizes " << mask_left_size << ", " << mask_right_size << '\n';
	  total_outside_scatter[bin.segment_num()][bin.axial_pos_num()] +=  
	    std::accumulate(scatter_sinogram[bin.view_num()].begin(),
			    scatter_sinogram[bin.view_num()].begin() + mask_left_size,
			    0.F) +
	    std::accumulate(scatter_sinogram[bin.view_num()].rbegin(),
			    scatter_sinogram[bin.view_num()].rbegin() + mask_right_size,
			    0.F);
	  total_outside_emission[bin.segment_num()][bin.axial_pos_num()] +=  
	    std::accumulate(emission_sinogram[bin.view_num()].begin(),
			    emission_sinogram[bin.view_num()].begin() + mask_left_size,
			    0.F) +
	    std::accumulate(emission_sinogram[bin.view_num()].rbegin(),
			    emission_sinogram[bin.view_num()].rbegin() + mask_right_size,
			    0.F);
	  count += mask_left_size + mask_right_size;
#endif
	  }
#ifndef NDEBUG
	std::cout << total_outside_emission[bin.segment_num()][bin.axial_pos_num()] << " " <<
	  total_outside_scatter[bin.segment_num()][bin.axial_pos_num()] << '\n';
#endif
	std::cout << count << " bins in mask\n";
	if (scatter_sinogram.sum()==0)
	  {  
	    scale_factors[bin.segment_num()][bin.axial_pos_num()] = 0;
	  }
	else
	  {
	    if (total_outside_scatter[bin.segment_num()][bin.axial_pos_num()]<=
		scatter_sinogram.sum()/
		(proj_data_info.get_num_views() * proj_data_info.get_num_tangential_poss()) * .001
		)
	      {
	    error("Problem in finding scatter scaling factor.\n"
		  "Scatter data in mask %g too small compared to total in sinogram %g.\n"
		  "Adjust threshold?",
		  total_outside_scatter[bin.segment_num()][bin.axial_pos_num()],
		  scatter_sinogram.sum()
		  );
	      }
	    scale_factors[bin.segment_num()][bin.axial_pos_num()] = 
	      total_outside_emission[bin.segment_num()][bin.axial_pos_num()]/
	      total_outside_scatter[bin.segment_num()][bin.axial_pos_num()];
	  }
      }

  {
    VectorWithOffset<float> kernel(-1,1);
    kernel.fill(1.F/3);
    ArrayFilter1DUsingConvolution<float> filter(kernel);
    for (Array<2,float>::iterator row_iter = scale_factors.begin();
	 row_iter != scale_factors.end();
	 ++row_iter)
      {
	Array<1,float> temp = *row_iter;
	const int min_index = temp.get_min_index();
	const int max_index = temp.get_max_index();
	temp.resize(min_index-1, max_index+1);
	temp[min_index-1] = temp[min_index];
	temp[max_index+1] = temp[max_index];
	filter(*row_iter, temp);
      }
  }
  return scale_factors;		
}
END_NAMESPACE_STIR
		
