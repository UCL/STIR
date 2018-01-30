//
//
/*
    Copyright (C) 2007- 2009, Hammersmith Imanet Ltd
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
\ingroup reconstructors
\brief  disabled implementation of doing a simple line search (from the stir::OSSPSReconstruction class)

\author Kris Thielemans
  
*/

#error this code does not compile right now

// only used in (disabled) line_search below
static
void accumulate_loglikelihood(RelatedViewgrams<float>& projection_data, 
			      const RelatedViewgrams<float>& estimated_projections,
			      float* accum)
{
  RelatedViewgrams<float>::iterator p_iter = projection_data.begin();
  RelatedViewgrams<float>::iterator p_end = projection_data.end();
  RelatedViewgrams<float>::const_iterator e_iter = estimated_projections.begin();
  while (p_iter!= p_end)
    {
      accumulate_loglikelihood(*p_iter, *e_iter, /* rim_truncation_sino*/0, accum);
      ++p_iter; ++e_iter;
    }
}
  
template <class TargetT>
float 
OSSPSReconstruction<TargetT>::
line_search(const TargetT& current_estimate, const TargetT& additive_update)
{

  if (!this->do_line_search)
    return 1;

#if 1
  error("OSSPS line search disabled for now");
  return 0;
#else
  float result;
            
  CPUTimer timer;
  timer.reset();
  timer.start();

  PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>&
    objective_function =
    static_cast<PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>&>
    (*this->objective_function_sptr);

  shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr =
    objective_function.get_projector_pair().get_symmetries_used()->clone();

  const double start_time =
    objective_function.get_time_frame_definitions().get_start_time(objective_function.get_time_frame_num());
  const double end_time =
    objective_function.get_time_frame_definitions().get_end_time(objective_function.get_time_frame_num());

  //for (int segment_num = -objective_function.get_max_segment_num_to_process();
  //     segment_num<= objective_function.get_max_segment_num_to_process();
  //     ++segment_num) 
  //     {      
  //    for (int view = objective_function.get_proj_data().get_min_view_num(); 
  //	   view <= objective_function.get_proj_data().get_max_view_num(); 
  //	   ++view)
  const int segment_num=0;
  const int view_num=0;

  {
    const ViewSegmentNumbers view_segment_num(view_num, segment_num);
    
    if (!symmetries_sptr->is_basic(view_segment_num))
      error("line search:seg 0, view 0 is not basic");

    RelatedViewgrams<float> measured_viewgrams =
      objective_function.get_proj_data().get_related_viewgrams(view_segment_num, symmetries_sptr);
    RelatedViewgrams<float> forward_model_viewgrams =
      measured_viewgrams.get_empty_copy();
    {
      objective_function.get_projector_pair().get_forward_projector_sptr()->
	forward_project(forward_model_viewgrams, current_estimate);
      
      objective_function.get_normalisation().undo(forward_model_viewgrams, start_time, end_time);
      
      const ProjData * const additive_ptr = 
	objective_function.get_additive_proj_data_sptr().get();
      if (!is_null_ptr(additive_ptr))
	forward_model_viewgrams +=
	  additive_ptr->get_related_viewgrams(view_segment_num, symmetries_sptr);
    }

    RelatedViewgrams<float> forward_model_update_viewgrams =
      measured_viewgrams.get_empty_copy();
    {
      objective_function.get_projector_pair().get_forward_projector_sptr()->
	forward_project(forward_model_update_viewgrams, additive_update);
    }

    RelatedViewgrams<float> tmp_viewgrams =
      measured_viewgrams.get_empty_copy();
    
#define FUNC(value, alpha) \
    {\
      value = 0; \
    tmp_viewgrams = forward_model_update_viewgrams; \
    tmp_viewgrams *= alpha; \
    tmp_viewgrams += forward_model_viewgrams; \
    accumulate_loglikelihood(measured_viewgrams,\
			     tmp_viewgrams, 	\
			     &value);\
    }

    // simple bisection
    float min_alpha=.2;// used to be .1
    float max_alpha=10;// used to be 1000
    int iter_num=0;
    float current = 1;
    float previous=-1; // set somewhere crazy such that we do at least 1 iteration

    float value_at_min;
    float value_at_max;
    float value_at_current;
    FUNC(value_at_min, min_alpha);
    FUNC(value_at_max, max_alpha);
    while (iter_num++ != 100 && fabs(previous/current -1)>.02)
      {
	previous = current;
	FUNC(value_at_current, current);
	info(boost::format("line search at %1% value %2%") % current % value_at_current);
	if (value_at_current <= value_at_min &&
	    value_at_max <= value_at_current)
	  {
	    min_alpha = current;
	    value_at_min = value_at_current;
	    current = (current + max_alpha)/2;
	  }
	else if (value_at_current <= value_at_max && value_at_min <= value_at_current)
	  {
	    max_alpha = current;
	    value_at_max = value_at_current;
	    current = (current + min_alpha)/2;
	  }
	else
	  {
	    if (value_at_current > value_at_max ||
		value_at_current > value_at_min)
	      {
	      const float scale=(value_at_min+value_at_max)/2;
	      if (std::fabs((value_at_current - value_at_max)/scale) < .01 ||
		  std::fabs((value_at_current - value_at_min)/scale) < .01)
		{
		  // it's probably just rounding error, so we have converged
		  break; // out of while
		}
	      warning("line search error. Function non-convex?\n"
		      "min %g (%g), curr %g (delta %g), max %g (delta %g)",
		    min_alpha, value_at_min,
		    current, value_at_current-value_at_min,
		    max_alpha, value_at_max-value_at_min
		    );
	      const float list_from = std::max(0.F,min_alpha-1);
	      const float list_to = std::min(100.F,max_alpha+1);
	      const float increment = (list_to - list_from)/100;
	      for (current=list_from; current<=list_to; current += increment)
		{
		  FUNC(value_at_current, current);
		  info(boost::format("%1% d %2%") % current % value_at_current-value_at_min);
		}
	      exit(EXIT_FAILURE);
	      }
	    // need to decide if we go left or right
	    float half_way_left = (current + min_alpha)/2;
	    float value_at_half_way_left;
	    FUNC(value_at_half_way_left, half_way_left);
	    if (value_at_half_way_left > value_at_current)
	      {
		// it has to be at the right
		min_alpha = current;
		value_at_min = value_at_current;
		current = (current + max_alpha)/2;
	      }
	    else
	      {
		// it's at the left.
		// TODO we'll be recomputing the value here
		max_alpha = current;
		value_at_max = value_at_current;
		current = (current + min_alpha)/2;
	      }
	  }
      } // end of while
    result = current;
  }
  timer.stop();
  info(boost::format("line search took %1% s CPU time") % timer.value());
  info(boost::format("value for alpha %1%") % result);
  /*
  this->output_file_format_ptr->
	write_to_file("curr_est", current_estimate);
  this->output_file_format_ptr->
	write_to_file("upd", additive_update);

  exit(0);
  */
  return result;
#endif
}
