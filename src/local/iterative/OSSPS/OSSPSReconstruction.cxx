//
// $Id$
//
/*
    Copyright (C) 2002- $Date$, Hammersmith Imanet Ltd
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
\ingroup OSSPS  
\ingroup reconstructors
\brief  implementation of the stir::OSSPSReconstruction class 

\author Sanida Mustafovic
\author Kris Thielemans
  
$Date$
$Revision$
*/

#include "local/stir/OSSPS/OSSPSReconstruction.h"
#include "stir/min_positive_element.h"
#include "stir/recon_array_functions.h"
#include "stir/recon_buildblock/PriorWithParabolicSurrogate.h"
#include "stir/Succeeded.h"
#include "stir/recon_array_functions.h"
#include "stir/thresholding.h"
#include "stir/is_null_ptr.h"
#include "stir/NumericInfo.h"
#include "stir/utilities.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMean.h"

#include <iostream>
#include <memory>
#include <iostream>
#include <algorithm>
#ifdef BOOST_NO_STRINGSTREAM
#include <strstream.h>
#else
#include <sstream>
#endif
#include "boost/lambda/lambda.hpp"

#ifndef STIR_NO_NAMESPACES
using std::auto_ptr;
using std::cerr;
using std::endl;
using boost::lambda::_1;
using boost::lambda::_2;
#endif


START_NAMESPACE_STIR

//*************** parameters *************


template <class TargetT>
void 
OSSPSReconstruction<TargetT>::
set_defaults()
{
  base_type::set_defaults();
  enforce_initial_positivity = 0;
  upper_bound = NumericInfo<float>().max_value();
  write_update_image = 0;
  precomputed_denominator_filename = "";

  //MAP_model="additive"; 
  relaxation_parameter = 1;
  relaxation_gamma = 0.1F;

  this->do_line_search = false;
}

template <class TargetT>
void 
OSSPSReconstruction<TargetT>::
initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("OSSPSParameters");
  this->parser.add_stop_key("End");
  
  this->parser.add_key("enforce initial positivity condition",&enforce_initial_positivity);
  //this->parser.add_key("MAP_model", &MAP_model);
  this->parser.add_key("upper bound", &upper_bound);
  this->parser.add_key("write update image",&write_update_image);   
  this->parser.add_key("precomputed denominator", &precomputed_denominator_filename);

  this->parser.add_key("relaxation parameter", &relaxation_parameter);
  this->parser.add_key("relaxation gamma", &relaxation_gamma);

  this->parser.add_key("do_line_search", &this->do_line_search);
}


template <class TargetT>
void 
OSSPSReconstruction<TargetT>::
ask_parameters()
{
  error("Currently incomplete code. Use a parameter file. Sorry.");

  base_type::ask_parameters();
  
  // KT 05/07/2000 made enforce_initial_positivity int
  enforce_initial_positivity=
    ask("Enforce initial positivity condition?",true) ? 1 : 0;
  
  char precomputed_denominator_filename_char[max_filename_length];
  
  ask_filename_with_extension(precomputed_denominator_filename_char,"Enter file name of precomputed denominator  (1 = 1's): ", "");   
  
  precomputed_denominator_filename=precomputed_denominator_filename_char;
  


 {       
    
    cerr<<endl<<"Supply objective function type:\nPossible values:\n";
    GeneralisedObjectiveFunction<TargetT>::list_registered_names(cerr); 
    const string objective_function_type = ask_string("");
    
    this->objective_function_sptr = 
      GeneralisedObjectiveFunction<TargetT>::read_registered_object(0, objective_function_type); 
    
  } 
  
  // KT 17/08/2000 3 new parameters
  const double max_in_double = static_cast<double>(NumericInfo<float>().max_value());
  upper_bound = ask_num("upper bound",
    1.,max_in_double,max_in_double);
  
  write_update_image = ask_num("write update image", 0,1,0);

  // TODO some more parameters here (relaxation et al)
}




template <class TargetT>
bool 
OSSPSReconstruction<TargetT>::
post_processing()
{
  if (base_type::post_processing())
    return true;
  return false;
}

//*************** other functions *************

template <class TargetT>
OSSPSReconstruction<TargetT>::
OSSPSReconstruction()
{  
  set_defaults();
}

template <class TargetT>
OSSPSReconstruction<TargetT>::
OSSPSReconstruction(const string& parameter_filename)
{    
  this->initialise(parameter_filename);
  cerr<<this->parameter_info();
}

template <class TargetT>
std::string 
OSSPSReconstruction<TargetT>::
method_info() const
{
  
  // TODO add prior name?
  
#ifdef BOOST_NO_STRINGSTREAM
  char str[10000];
  ostrstream s(str, 10000);
#else
  std::ostringstream s;
#endif
  
  // if(inter_update_filter_interval>0) s<<"IUF-";
  if (!this->objective_function_sptr->prior_is_zero())
    s << "MAP-";
  if(this->num_subsets>1) 
    s<<"OS-";
  s << "SPS";
  if(this->inter_iteration_filter_interval>0) s<<"S";  

  return s.str();
}

template <class TargetT>
Succeeded 
OSSPSReconstruction<TargetT>::
precompute_denominator_of_conditioner_without_penalty()
{
            
  CPUTimer timer;
  timer.reset();
  timer.start();

  assert(*std::max_element(precomputed_denominator_ptr->begin_all(), precomputed_denominator_ptr->end_all()) == 0);
  assert(*std::min_element(precomputed_denominator_ptr->begin_all(), precomputed_denominator_ptr->end_all()) == 0);

  // TODO replace by boost::scoped_ptr
  std::auto_ptr<TargetT > data_full_of_ones_aptr =
	std::auto_ptr<TargetT >
	( precomputed_denominator_ptr->clone());
  std::fill(data_full_of_ones_aptr->begin_all(),
	    data_full_of_ones_aptr->end_all(),
	    1);

  this->objective_function_sptr->
    add_multiplication_with_approximate_Hessian_without_penalty(
								*precomputed_denominator_ptr, 
								*data_full_of_ones_aptr);
  timer.stop();
  cerr << "Precomputing denominator took " << timer.value() << " s CPU time\n";
  cerr << "min and max in precomputed denominator " 
       << *std::min_element(precomputed_denominator_ptr->begin_all(), precomputed_denominator_ptr->end_all())
       << ", "
       << *std::max_element(precomputed_denominator_ptr->begin_all(), precomputed_denominator_ptr->end_all())
       << std::endl;
  

  // Write it to file
  {
    std::string fname =  
      this->output_filename_prefix +
      "_precomputed_denominator";
    
      cerr <<"  - Saving " << fname << endl;
      this->output_file_format_ptr->
	write_to_file(fname, *precomputed_denominator_ptr);
  }

  return Succeeded::yes;
}

#if 0
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
#endif

  
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
	std::cerr << "line search at " << current << " value " << value_at_current << '\n';
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
		  std::cerr << current << " d " << value_at_current-value_at_min << '\n';
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
  std::cerr << "line search  took " << timer.value() << " s CPU time\n";
  std::cerr << "value for alpha " << result << '\n';
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

template <class TargetT>
Succeeded 
OSSPSReconstruction<TargetT>::
set_up(shared_ptr <TargetT > const& target_image_ptr)
{
  if (base_type::set_up(target_image_ptr) == Succeeded::no)
    return Succeeded::no;

  if (this->relaxation_parameter<=0)
    {
      warning("OSSPS: relaxation parameter should be positive but is %g",
	      this->relaxation_parameter);
      return Succeeded::no;
    }
  if (this->relaxation_gamma<0)
    {
      warning("OSSPS: relaxation_gamma parameter should be non-negative but is %g",
	      this->relaxation_gamma);
      return Succeeded::no;
    }

  if (!is_null_ptr(this->get_prior_ptr())&& 
      dynamic_cast<PriorWithParabolicSurrogate<TargetT>*>(this->get_prior_ptr())==0)
  {
    warning("OSSPS: Prior must be of a type derived from PriorWithParabolicSurrogate\n");
    return Succeeded::no;
  }

  if(enforce_initial_positivity) 
    threshold_min_to_small_positive_value(target_image_ptr->begin_all(),
					  target_image_ptr->end_all(),
					  10.E-6F);
  
  if(this->precomputed_denominator_filename=="")
  {
    precomputed_denominator_ptr=target_image_ptr->get_empty_copy();
    precompute_denominator_of_conditioner_without_penalty();
  }
  else if(this->precomputed_denominator_filename=="1")
  {
    precomputed_denominator_ptr=target_image_ptr->get_empty_copy();
    std::fill(precomputed_denominator_ptr->begin_all(), precomputed_denominator_ptr->end_all(), 1.F);
  }
  else
  {       
    precomputed_denominator_ptr = 
      TargetT::read_from_file(this->precomputed_denominator_filename);   
    {
      string explanation;
      if (!precomputed_denominator_ptr->has_same_characteristics(*target_image_ptr, explanation))
	{
	  warning("OSSPS: precomputed_denominator should have same characteristics as target image: %s",
		  explanation.c_str());
	  return Succeeded::no;
	}    
    }
  }  
  return Succeeded::yes;
}



/*! \brief OSSPS additive update at every subiteration
  \warning This modifies *precomputed_denominator_ptr. So, you <strong>have to</strong>
  call set_up() before running a new reconstruction.
  */
template <class TargetT>
void 
OSSPSReconstruction<TargetT>::
update_estimate(TargetT &current_image_estimate)
{
  if (this->get_subiteration_num() == this->get_start_subiteration_num())
    {
      // set all voxels to 0 that cannot be estimated.
      this->objective_function_sptr->
        fill_nonidentifiable_target_parameters(current_image_estimate, 0);
    }
  // Check if we need to recompute the penalty term in the denominator during iterations .
  // For the quadratic prior, this is independent of the image (only on kappa's)
  // And of course, it's also independent when there is no prior
  // TODO by default, this should be off probably (to save time).
  const bool recompute_penalty_term_in_denominator =
    !this->objective_function_sptr->prior_is_zero() &&
    static_cast<PriorWithParabolicSurrogate<TargetT> const&>(*this->get_prior_ptr()).
     parabolic_surrogate_curvature_depends_on_argument();
#ifndef PARALLEL
  //CPUTimer subset_timer;
  //subset_timer.start();
#else // PARALLEL
  PTimer timerSubset;
  timerSubset.Start();
#endif // PARALLEL
  
  const int subset_num=this->get_subset_num();  
  cerr<<endl<<"Now processing subset #: "<<subset_num<<endl;
    
  // TODO make member or static parameter to avoid reallocation all the time
  auto_ptr< TargetT > numerator_ptr =
    auto_ptr< TargetT >(current_image_estimate.get_empty_copy());

  this->objective_function_sptr->compute_sub_gradient(*numerator_ptr, current_image_estimate, subset_num);
  //*numerator_ptr *= this->num_subsets;
  std::transform(numerator_ptr->begin_all(), numerator_ptr->end_all(),
		 numerator_ptr->begin_all(),
		 _1 * this->num_subsets);

  cerr<< "num subsets " << this->num_subsets << '\n';  
  cerr << "this->num_subsets*subgradient : max " 
       << *std::max_element(numerator_ptr->begin_all(), numerator_ptr->end_all());
  cerr << ", min " 
       << *std::min_element(numerator_ptr->begin_all(), numerator_ptr->end_all())
       << endl;

  // now divide by denominator

  if (recompute_penalty_term_in_denominator || 
      (this->get_subiteration_num() == this->get_start_subiteration_num()))
    {
      auto_ptr< TargetT > work_image_ptr = 
	auto_ptr< TargetT >(current_image_estimate.get_empty_copy());
      
      // avoid work (or crash) when penalty is 0
      if (!this->objective_function_sptr->prior_is_zero())
	{
	  static_cast<PriorWithParabolicSurrogate<TargetT>&>(*get_prior_ptr()).
	    parabolic_surrogate_curvature(*work_image_ptr, current_image_estimate);   
	  //*work_image_ptr *= 2;
	  //*work_image_ptr += *precomputed_denominator_ptr ;
	  std::transform(work_image_ptr->begin_all(), work_image_ptr->end_all(),
			 precomputed_denominator_ptr->begin_all(), 
			 work_image_ptr->begin_all(),
			 _1 * 2 + _2);
	}
      else
	*work_image_ptr = *precomputed_denominator_ptr ;
    
      // KT 09/12/2002 new
      // avoid division by 0 by thresholding the denominator to be strictly positive      
      threshold_min_to_small_positive_value(work_image_ptr->begin_all(),
					    work_image_ptr->end_all(),
					    10.E-6F);
      cerr << " denominator max " 
	   << *std::max_element(work_image_ptr->begin_all(), work_image_ptr->end_all());
      cerr << ", min " 
	   << *std::min_element(work_image_ptr->begin_all(), work_image_ptr->end_all())
	   << endl;

      if (!recompute_penalty_term_in_denominator)
	{
	  // store for future use
	  *precomputed_denominator_ptr = *work_image_ptr;
	}
    
      //*numerator_ptr /= *work_image_ptr;
      std::transform(numerator_ptr->begin_all(), numerator_ptr->end_all(),
		     work_image_ptr->begin_all(),
		     numerator_ptr->begin_all(), 
		     _1 / _2);

    }
  else
    {
      // we have computed the denominator already 
      //*numerator_ptr /= *precomputed_denominator_ptr;
      std::transform(numerator_ptr->begin_all(), numerator_ptr->end_all(),
		     precomputed_denominator_ptr->begin_all(),
		     numerator_ptr->begin_all(), 
		     _1 / _2);

    }

  //relaxation_parameter ~1/(1+n) where n is iteration number 
  const float relaxation_parameter = this->relaxation_parameter/
    (1+this->relaxation_gamma*(this->subiteration_num/this->num_subsets));


  std::cerr << "relaxation parameter = " << relaxation_parameter << '\n';

  const float alpha =
    line_search(current_image_estimate, *numerator_ptr);
  // *numerator_ptr *= relaxation_parameter * alpha;  
  std::transform(numerator_ptr->begin_all(), numerator_ptr->end_all(),
		 numerator_ptr->begin_all(),
		 _1 * relaxation_parameter * alpha);

  
  if (write_update_image)
    {
      // Write it to file
      const std::string fname =
	this->make_filename_prefix_subiteration_num(this->output_filename_prefix + "_update");
      this->output_file_format_ptr->
	write_to_file(fname, *numerator_ptr);
    }
  
  {
    cerr << "additive update image min,max: " 
	 << *std::min_element(numerator_ptr->begin_all(), numerator_ptr->end_all())
	 << ", " 
	 << *std::max_element(numerator_ptr->begin_all(), numerator_ptr->end_all())
	 << endl;
  }  
  current_image_estimate += *numerator_ptr; 

  // now threshold image
  {
    const float current_min =
      *std::min_element(current_image_estimate.begin_all(),
			current_image_estimate.end_all()); 
    const float current_max = 
      *std::max_element(current_image_estimate.begin_all(),
			current_image_estimate.end_all()); 
    const float new_min = 0.F;
    const float new_max = 
      static_cast<float>(upper_bound);
    cerr << "current image old min,max: " 
	 << current_min
	 << ", " 
	 << current_max
	 << ", new min,max " 
	 << max(current_min, new_min) << ", " << min(current_max, new_max)
	 << endl;
    
    threshold_upper_lower(current_image_estimate.begin_all(),
			  current_image_estimate.end_all(), 
			  new_min, new_max);      
  }  

#ifndef PARALLEL
  //cerr << "Subset : " << subset_timer.value() << "secs " <<endl;
#else // PARALLEL
  timerSubset.Stop();
  cerr << "Subset: " << timerSubset.GetTime() << "secs" << endl;

#endif
  
}
END_NAMESPACE_STIR


///////// instantiations
#include "stir/DiscretisedDensity.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
START_NAMESPACE_STIR

template class OSSPSReconstruction<DiscretisedDensity<3,float> >;
template class OSSPSReconstruction<ParametricVoxelsOnCartesianGrid >; 

END_NAMESPACE_STIR


