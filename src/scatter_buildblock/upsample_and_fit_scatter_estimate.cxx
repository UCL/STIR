/*
  Copyright (C) 2005 - 2011-12-31, Hammersmith Imanet Ltd
  Copyright (C) 2014, University College London
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
\ingroup scatter
\brief implementation of stir::ScatterEstimationByBin::upsample_and_fit_scatter_estimate

\author Charalampos Tsoumpas
\author Kris Thielemans
*/

#include "stir/ProjDataInfo.h"
#include "stir/ExamInfo.h"
#include "stir/ProjDataInMemory.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/Viewgram.h"
#include "stir/inverse_SSRB.h"
#include "stir/scale_sinograms.h"
#include "stir/scatter/ScatterEstimation.h"
#include "stir/recon_buildblock/BinNormalisation.h"
#include "stir/interpolate_projdata.h"
#include "stir/utilities.h"
#include "stir/IndexRange2D.h" 
#include "stir/stream.h"
#include "stir/Succeeded.h"
#include "stir/thresholding.h"
#include "stir/is_null_ptr.h"
#include "stir/ArrayFilter1DUsingConvolution.h"
#include <iostream>
#include <fstream>
#include <string>
/***********************************************************/

START_NAMESPACE_STIR

void 
ScatterEstimation::
upsample_and_fit_scatter_estimate(ProjData& scaled_scatter_proj_data,
                                  const  ProjData& emission_proj_data,
                                  const ProjData& scatter_proj_data,
                                  BinNormalisation& scatter_normalisation,
                                  const ProjData& weights_proj_data,
                                  const float min_scale_factor,
                                  const float max_scale_factor,
                                  const unsigned half_filter_width,
                                  BSpline::BSplineType spline_type,
                                  const bool remove_interleaving)
{

    shared_ptr<ProjDataInfo> interpolated_direct_scatter_proj_data_info_sptr(emission_proj_data.get_proj_data_info_ptr()->clone());
    interpolated_direct_scatter_proj_data_info_sptr->reduce_segment_range(0,0);


    info("upsample_and_fit_scatter_estimate: Interpolating scatter estimate to size of emission data");
    ProjDataInMemory interpolated_direct_scatter(emission_proj_data.get_exam_info_sptr(),
					       interpolated_direct_scatter_proj_data_info_sptr);        

    interpolate_projdata(interpolated_direct_scatter, scatter_proj_data, spline_type, remove_interleaving);


    const TimeFrameDefinitions& time_frame_defs = emission_proj_data.get_exam_info_sptr()->time_frame_definitions;

    if (min_scale_factor != 1 || max_scale_factor != 1 || !scatter_normalisation.is_trivial())
    {
      ProjDataInMemory interpolated_scatter(emission_proj_data.get_exam_info_sptr(),
					    emission_proj_data.get_proj_data_info_ptr()->create_shared_clone());
      inverse_SSRB(interpolated_scatter, interpolated_direct_scatter);

      scatter_normalisation.set_up(emission_proj_data.get_proj_data_info_ptr()->create_shared_clone());
      scatter_normalisation.undo(interpolated_scatter, 
                                 time_frame_defs.get_start_time(), time_frame_defs.get_end_time());
      Array<2,float> scale_factors;
      
      info("upsample_and_fit_scatter_estimate: Finding scale factors by sinogram");
      scale_factors = get_scale_factors_per_sinogram(
                                                 emission_proj_data, 
                                                 interpolated_scatter,
                                                 weights_proj_data);
    
      std::cout << scale_factors;
      threshold_lower(scale_factors.begin_all(), 
                      scale_factors.end_all(),
                      min_scale_factor);
      threshold_upper(scale_factors.begin_all(), 
                      scale_factors.end_all(),
                      max_scale_factor);
      info("upsample_and_fit_scatter_estimate: After thresholding:");
      std::cout << scale_factors;
      VectorWithOffset<float> kernel(-static_cast<int>(half_filter_width),half_filter_width);
      kernel.fill(1.F/(2*half_filter_width+1));
      ArrayFilter1DUsingConvolution<float> lowpass_filter(kernel, BoundaryConditions::constant);
      std::for_each(scale_factors.begin(), 
                    scale_factors.end(),
                    lowpass_filter);
      info("upsample_and_fit_scatter_estimate: After filtering:");
      std::cout << scale_factors;
      info("upsample_and_fit_scatter_estimate: applying scale factors");
      if (scale_sinograms(scaled_scatter_proj_data, 
                          interpolated_scatter,
                          scale_factors) != Succeeded::yes)
        {
          error("upsample_and_fit_scatter_estimate: writing of scaled sinograms failed");
        }
    }
  else // min/max_scale_factor equal to 1
    {
      inverse_SSRB(scaled_scatter_proj_data, interpolated_direct_scatter);
    }
}


void
ScatterEstimation::
upsample_scatter_estimate(ProjData& scaled_scatter_proj_data,
                                  const  ProjData& emission_proj_data,
                                  const ProjData& scatter_proj_data,
                                  const bool remove_interleaving)
{
    stir::BSpline::BSplineType  spline_type = stir::BSpline::linear;
    shared_ptr<ProjDataInfo> interpolated_direct_scatter_proj_data_info_sptr(emission_proj_data.get_proj_data_info_ptr()->clone());
    interpolated_direct_scatter_proj_data_info_sptr->reduce_segment_range(0,0);


    info("upsample_and_fit_scatter_estimate: Interpolating scatter estimate to size of emission data");
    ProjDataInMemory interpolated_direct_scatter(emission_proj_data.get_exam_info_sptr(),
                           interpolated_direct_scatter_proj_data_info_sptr);

    // interpolate projdata
    interpolate_projdata(interpolated_direct_scatter, scatter_proj_data, spline_type, remove_interleaving);

    // Perform Inverse Single Slice Rebinning
    inverse_SSRB(scaled_scatter_proj_data, interpolated_direct_scatter);

}



void
ScatterEstimation::
pull_scatter_estimate(ProjData& scaled_scatter_proj_data,
                                  const  ProjData& emission_proj_data,
                                  const ProjData& scatter_proj_data,
                                  const bool remove_interleaving)
{

    shared_ptr<ProjDataInfo> interpolated_direct_scatter_proj_data_info_sptr(emission_proj_data.get_proj_data_info_ptr()->clone());
    interpolated_direct_scatter_proj_data_info_sptr->reduce_segment_range(0,0); //create the output template


    info("upsample_and_fit_scatter_estimate: Interpolating scatter estimate to size of emission data");
    ProjDataInMemory interpolated_direct_scatter(emission_proj_data.get_exam_info_sptr(),
                           interpolated_direct_scatter_proj_data_info_sptr);

    // interpolate projdata
    interpolate_projdata_pull(interpolated_direct_scatter, scatter_proj_data, remove_interleaving);

    // Perform Inverse Single Slice Rebinning
    inverse_SSRB(scaled_scatter_proj_data, interpolated_direct_scatter);

}

void
ScatterEstimation::
push_scatter_estimate(ProjData& scaled_scatter_proj_data,
                                  const  ProjData& emission_proj_data,
                                  const ProjData& scatter_proj_data,
                                  const bool remove_interleaving)
{

    shared_ptr<ProjDataInfo> new_input_proj_data_info_sptr(scatter_proj_data.get_proj_data_info_ptr()->clone());
    new_input_proj_data_info_sptr->reduce_segment_range(0,0); //create input template

    ProjDataInMemory new_input(scatter_proj_data.get_exam_info_sptr(),new_input_proj_data_info_sptr);
    transpose_inverse_SSRB(new_input, scatter_proj_data);


    interpolate_projdata_push(scaled_scatter_proj_data, new_input, remove_interleaving);


}


void
ScatterEstimation::
pull_scatter_estimate(ProjData& scaled_scatter_proj_data,
                                  const  ProjData& emission_proj_data,
                                  const ProjData& scatter_proj_data,
                                  const ProjData& norm,
                                  const bool remove_interleaving)
{

    shared_ptr<ProjDataInfo> interpolated_direct_scatter_proj_data_info_sptr(emission_proj_data.get_proj_data_info_ptr()->clone());
    interpolated_direct_scatter_proj_data_info_sptr->reduce_segment_range(0,0); //create the output template


    info("upsample_and_fit_scatter_estimate: Interpolating scatter estimate to size of emission data");
    ProjDataInMemory interpolated_direct_scatter(emission_proj_data.get_exam_info_sptr(),
                           interpolated_direct_scatter_proj_data_info_sptr);

    // interpolate projdata
    interpolate_projdata_pull(interpolated_direct_scatter, scatter_proj_data, remove_interleaving);

    // Perform Inverse Single Slice Rebinning
    inverse_SSRB(scaled_scatter_proj_data, interpolated_direct_scatter);

    apply_norm(scaled_scatter_proj_data,norm);

}

void
ScatterEstimation::
push_scatter_estimate(ProjData& scaled_scatter_proj_data,
                                  const  ProjData& emission_proj_data,
                                  const ProjData& scatter_proj_data,
                                   const ProjData& norm,
                                  const bool remove_interleaving)
{

    ProjDataInMemory scatter_proj_data_in_memory(scatter_proj_data);
    apply_norm(scatter_proj_data_in_memory,norm);

    shared_ptr<ProjDataInfo> new_input_proj_data_info_sptr(scatter_proj_data_in_memory.get_proj_data_info_ptr()->clone());
    new_input_proj_data_info_sptr->reduce_segment_range(0,0); //create input template

    ProjDataInMemory new_input(scatter_proj_data_in_memory.get_exam_info_sptr(),new_input_proj_data_info_sptr);

    transpose_inverse_SSRB(new_input, scatter_proj_data_in_memory);


    interpolate_projdata_push(scaled_scatter_proj_data, new_input, remove_interleaving);


}

void
ScatterEstimation::
apply_norm(ProjData& projdata,const ProjData& norm)
{

if((projdata.get_num_views()!=norm.get_num_views())||(projdata.get_num_tangential_poss()!=norm.get_num_tangential_poss()))
    error("sinograms have to have the same dimensions");
ProjDataInMemory projdata_out(projdata);
projdata_out.fill(0);

ViewSegmentNumbers vs_num;

for (vs_num.segment_num() = norm.get_min_segment_num(); vs_num.segment_num() <= norm.get_max_segment_num(); ++vs_num.segment_num())
{
    for (vs_num.view_num() = norm.get_min_view_num();vs_num.view_num() <= norm.get_max_view_num(); ++vs_num.view_num())
    {

        Viewgram<float> viewgram_n = norm.get_viewgram(vs_num.view_num(), vs_num.segment_num());
        Viewgram<float> viewgram_in = projdata.get_viewgram(vs_num.view_num(), vs_num.segment_num());
        viewgram_in *= viewgram_n;
        projdata_out.set_viewgram(viewgram_in);
    }
}

projdata.fill(projdata_out);

}
END_NAMESPACE_STIR
