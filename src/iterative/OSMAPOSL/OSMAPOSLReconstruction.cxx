//
// $Id$
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/

/*!
  \file
  \ingroup OSMAPOSL  
  \ingroup reconstructors
  \brief  implementation of the OSMAPOSLReconstruction class 
    
  \author Matthew Jacobson
  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project
      
  $Date$
  $Revision$
*/

#include "stir/OSMAPOSL/OSMAPOSLReconstruction.h"
#include "stir/recon_array_functions.h"
#include "stir/DiscretisedDensity.h"
#include "stir/LogLikBased/common.h"
#include "stir/ThresholdMinToSmallPositiveValueImageProcessor.h"
#include "stir/ChainedImageProcessor.h"
#include "stir/Succeeded.h"
#include "stir/thresholding.h"
#include "stir/is_null_ptr.h"
#include "stir/NumericInfo.h"
#include "stir/utilities.h"
// for get_symmetries_ptr()
#include "stir/DataSymmetriesForViewSegmentNumbers.h"
#include "stir/ViewSegmentNumbers.h"

#include <memory>
#include <iostream>
#ifdef BOOST_NO_STRINGSTREAM
#include <strstream.h>
#else
#include <sstream>
#endif

#ifndef STIR_NO_NAMESPACES
using std::auto_ptr;
using std::cerr;
using std::ends;
using std::endl;
#endif


START_NAMESPACE_STIR

//*********** parameters ***********

void 
OSMAPOSLReconstruction::set_defaults()
{
  LogLikelihoodBasedReconstruction::set_defaults();
  enforce_initial_positivity = true;
  do_rim_truncation = true;
  // KT 17/08/2000 3 new parameters
  maximum_relative_change = NumericInfo<float>().max_value();
  minimum_relative_change = 0;
  write_update_image = 0;
  inter_update_filter_interval = 0;
  inter_update_filter_ptr = 0;
  MAP_model="additive"; 
  prior_ptr = 0;
}

void
OSMAPOSLReconstruction::initialise_keymap()
{
  LogLikelihoodBasedReconstruction::initialise_keymap();
  parser.add_start_key("OSMAPOSLParameters");
  parser.add_stop_key("End");

  parser.add_key("enforce initial positivity condition",&enforce_initial_positivity);
  parser.add_key("do_rim_truncation",&do_rim_truncation);
  parser.add_key("inter-update filter subiteration interval",&inter_update_filter_interval);
  //add_key("inter-update filter type", KeyArgument::ASCII, &inter_update_filter_type);
  parser.add_parsing_key("inter-update filter type", &inter_update_filter_ptr);
  parser.add_parsing_key("Prior type", &prior_ptr);
  parser.add_key("MAP_model", &MAP_model);
  parser.add_key("maximum relative change", &maximum_relative_change);
  parser.add_key("minimum relative change",&minimum_relative_change);
  parser.add_key("write update image",&write_update_image);   
}


void OSMAPOSLReconstruction::ask_parameters()
{

  LogLikelihoodBasedReconstruction::ask_parameters();

  enforce_initial_positivity=
    ask("Enforce initial positivity condition?",true);

  do_rim_truncation  =
    ask("Do rim truncation to curcilar FOV?",true);

  inter_update_filter_interval=
    ask_num("Do inter-update filtering at sub-iteration intervals of: ",0, num_subiterations, 0);
     
  if(inter_update_filter_interval>0)
  {       
    
    cerr<<endl<<"Supply inter-update filter type:\nPossible values:\n";
    ImageProcessor<3,float>::list_registered_names(cerr);
    
    const string inter_update_filter_type = ask_string("");
    
    inter_update_filter_ptr = 
      ImageProcessor<3,float>::read_registered_object(0, inter_update_filter_type);      
    
  } 

 if(ask("Include prior?",false))
  {       
    
    cerr<<endl<<"Supply prior type:\nPossible values:\n";
    GeneralisedPrior<float>::list_registered_names(cerr);
    
    const string prior_type = ask_string("");
    
    prior_ptr = 
      GeneralisedPrior<float>::read_registered_object(0, prior_type); 
    
    MAP_model = 
      ask_string("Use additive or multiplicative form of MAP-OSL ('additive' or 'multiplicative')","additive");

    
  } 
  
  // KT 17/08/2000 3 new parameters
  const double max_in_double = static_cast<double>(NumericInfo<float>().max_value());
  maximum_relative_change = ask_num("maximum relative change",
      1.,max_in_double,max_in_double);
  minimum_relative_change = ask_num("minimum relative change",
      0.,1.,0.);
  
  write_update_image = ask_num("write update image", 0,1,0);

}




bool OSMAPOSLReconstruction::post_processing()
{
  if (LogLikelihoodBasedReconstruction::post_processing())
    return true;

  if (!is_null_ptr(prior_ptr))
  {
    if (MAP_model != "additive" && MAP_model != "multiplicative")
    {
      warning("MAP model should have as value 'additive' or 'multiplicative', while it is '%s'\n",
	MAP_model.c_str());
      return true;
    }
  }
  return false;
}

//*********** other functions ***********



OSMAPOSLReconstruction::
OSMAPOSLReconstruction()
{  
  set_defaults();
}

OSMAPOSLReconstruction::
OSMAPOSLReconstruction(const string& parameter_filename)
{  
  initialise(parameter_filename);
  cerr<<parameter_info();
}

string OSMAPOSLReconstruction::method_info() const
{

  // TODO add prior name?

#ifdef BOOST_NO_STRINGSTREAM
  // dangerous for out-of-range, but 'old-style' ostrstream seems to need this
  char str[10000];
  ostrstream s(str, 10000);
#else
  std::ostringstream s;
#endif

  if(get_parameters().inter_update_filter_interval>0) s<<"IUF-";
  if(get_parameters().num_subsets>1) s<<"OS";
  if (get_parameters().prior_ptr == 0 || 
      get_parameters().prior_ptr->get_penalisation_factor() == 0)
    s<<"EM";
  else
    s << "MAPOSL";
  if(get_parameters().inter_iteration_filter_interval>0) s<<"S";

  return s.str();

}

void OSMAPOSLReconstruction::recon_set_up(shared_ptr <DiscretisedDensity<3,float> > const& target_image_ptr)
{
  LogLikelihoodBasedReconstruction::recon_set_up(target_image_ptr);

  if (get_parameters().max_segment_num_to_process==-1)
    get_parameters().max_segment_num_to_process =
      get_parameters().proj_data_ptr->get_max_segment_num();

  // check subset balancing
  {
    const DataSymmetriesForViewSegmentNumbers& symmetries =
      *projector_pair_ptr->get_back_projector_sptr()->get_symmetries_used();

    Array<1,int> num_vs_in_subset(num_subsets);
    num_vs_in_subset.fill(0);
    for (int subset_num=0; subset_num<num_subsets; ++subset_num)
      {
	for (int segment_num = -get_parameters().max_segment_num_to_process; 
	     segment_num <= get_parameters().max_segment_num_to_process; 
	     ++segment_num)
	  for (int view_num = get_parameters().proj_data_ptr->get_min_view_num() + subset_num; 
	       view_num <= get_parameters().proj_data_ptr->get_max_view_num(); 
	       view_num += num_subsets)
	    {
	      const ViewSegmentNumbers view_segment_num(view_num, segment_num);
	      if (!symmetries.is_basic(view_segment_num))
		continue;
	      num_vs_in_subset[subset_num] +=
		symmetries.num_related_view_segment_numbers(view_segment_num);
	    }
      }
    for (int subset_num=1; subset_num<num_subsets; ++subset_num)
      {
	if(num_vs_in_subset[subset_num] != num_vs_in_subset[0])
	  { 
	    error("Number of subsets is such that subsets will be very unbalanced.\n"
		  "OSMAPOSL cannot handle this.\n"
		  "Either reduce the number of symmetries used by the projector, or\n"
		  "change the number of subsets. It usually should be a divisor of\n"
		  "  %d/4 (or if that's not an integer, a divisor of %d/2).\n",
		  proj_data_ptr->get_num_views(),
		  proj_data_ptr->get_num_views());
	  }
      }
  } // end check balancing


  if(get_parameters().enforce_initial_positivity) 
    threshold_min_to_small_positive_value(*target_image_ptr);

  if (inter_update_filter_interval<0)
    { error("Range error in inter-update filter interval");  }
  
  if(get_parameters().inter_update_filter_interval>0 && 
     !is_null_ptr(get_parameters().inter_update_filter_ptr))
    {
      // ensure that the result image of the filter is positive
      get_parameters().inter_update_filter_ptr =
	new ChainedImageProcessor<3,float>(
				  get_parameters().inter_update_filter_ptr,
				  new  ThresholdMinToSmallPositiveValueImageProcessor<float>);
      // KT 04/06/2003 moved set_up after chaining the filter. Otherwise it would be 
      // called again later on anyway.
      // Note however that at present, 
      cerr<<endl<<"Building inter-update filter kernel"<<endl;
      if (get_parameters().inter_update_filter_ptr->set_up(*target_image_ptr)
          == Succeeded::no)
	error("Error building inter-update filter\n");

    }
  if (get_parameters().inter_iteration_filter_interval>0 && 
      !is_null_ptr(get_parameters().inter_iteration_filter_ptr))
    {
      // ensure that the result image of the filter is positive
      get_parameters().inter_iteration_filter_ptr =
	new ChainedImageProcessor<3,float>(
					   get_parameters().inter_iteration_filter_ptr,
					   new  ThresholdMinToSmallPositiveValueImageProcessor<float>
);
      // KT 04/06/2003 moved set_up after chaining the filter (and removed it from IterativeReconstruction)
      cerr<<endl<<"Building inter-iteration filter kernel"<<endl;
      if (get_parameters().inter_iteration_filter_ptr->set_up(*target_image_ptr)
          == Succeeded::no)
	error("Error building inter iteration filter\n");
    }
}




void OSMAPOSLReconstruction::update_image_estimate(DiscretisedDensity<3,float> &current_image_estimate)
{
  
#ifndef PARALLEL
  //CPUTimer subset_timer;
  //subset_timer.start();
#else // PARALLEL
  PTimer timerSubset;
  timerSubset.Start();
#endif // PARALLEL
  
  // TODO make member parameter to avoid reallocation all the time
  auto_ptr< DiscretisedDensity<3,float> > multiplicative_update_image_ptr =
    auto_ptr< DiscretisedDensity<3,float> >(current_image_estimate.get_empty_discretised_density());
#if 1
  
  //For ordered set processing
  
  static VectorWithOffset<int> subset_array(get_parameters().num_subsets);
  
  if(get_parameters().randomise_subset_order && (subiteration_num-1)%get_parameters().num_subsets==0)
  {
    subset_array = randomly_permute_subset_order();
    
    cerr<<endl<<"Content ver.:"<<endl;
    
    for(int i=subset_array.get_min_index();i<=subset_array.get_max_index();i++) cerr<<subset_array[i]<<" ";
  };
  
  const int subset_num=get_parameters().randomise_subset_order ? subset_array[(subiteration_num-1)%get_parameters().num_subsets] : (subiteration_num+get_parameters().start_subset_num-1)%get_parameters().num_subsets;
  
  cerr<<endl<<"Now processing subset #: "<<subset_num<<endl;
  
  
  // KT 05/07/2000 made zero_seg0_end_planes int
  distributable_compute_gradient(*multiplicative_update_image_ptr, 
    current_image_estimate, 
    get_parameters().proj_data_ptr, 
    subset_num, 
    get_parameters().num_subsets, 
    -get_parameters().max_segment_num_to_process, // KT 30/05/2002 use new convention of distributable_* functions
    get_parameters().max_segment_num_to_process, 
    get_parameters().zero_seg0_end_planes!=0, 
    NULL, 
    additive_projection_data_ptr);
  
#endif  
  // divide by sensitivity  
  {
    // KT 05/11/98 clean now
    int count = 0;
    
    //std::cerr <<get_parameters().MAP_model << std::endl;
    
    if(get_parameters().prior_ptr == 0 || get_parameters().prior_ptr->get_penalisation_factor() == 0)     
    {
      if (do_rim_truncation)
	divide_and_truncate(*multiplicative_update_image_ptr, 
			    *sensitivity_image_ptr, 
			    rim_truncation_image,
			    count);
      else
	divide_array(*multiplicative_update_image_ptr, 
		     *sensitivity_image_ptr);
	
    }
    else
    {
      auto_ptr< DiscretisedDensity<3,float> > denominator_ptr = 
        auto_ptr< DiscretisedDensity<3,float> >(current_image_estimate.get_empty_discretised_density());
      
      
      get_parameters().prior_ptr->compute_gradient(*denominator_ptr, current_image_estimate); 
      
      if(get_parameters().MAP_model =="additive" )
      {
        // lambda_new = lambda / ((p_v + beta*prior_gradient)/ num_subsets) *
	//                   sum_subset backproj(measured/forwproj(lambda))
	// with p_v = sum_b p_bv
	// actually, we restrict 1 + beta*prior_gradient/p_v between .1 and 10

	for (int z=denominator_ptr->get_min_index();z<= denominator_ptr->get_max_index();z++)
	  for (int y=(*denominator_ptr)[z].get_min_index();y<= (*denominator_ptr)[z].get_max_index();y++)
	    for (int x=(*denominator_ptr)[z][y].get_min_index();x<= (*denominator_ptr)[z][y].get_max_index();x++)
	    {
	      const float sensitivity_at_voxel = (*sensitivity_image_ptr)[z][y][x];
	      float& denominator_at_voxel = (*denominator_ptr)[z][y][x];
	      const float sum = sensitivity_at_voxel + denominator_at_voxel;
	      // bound denominator between sensitivity_at_voxel/10 and sensitivity_at_voxel*10
	      denominator_at_voxel =
		max(min(sum, 10*sensitivity_at_voxel),sensitivity_at_voxel/10);
	    }
	    
      }
      else
      {
	if(get_parameters().MAP_model =="multiplicative" )
	{
	  // multiplicative form
	  // lambda_new = lambda / (p_v*(1 + beta*prior_gradient)/ num_subsets) *
	  //                   sum_subset backproj(measured/forwproj(lambda))
	  // with p_v = sum_b p_bv
	  // actually, we restrict 1 + beta*prior_gradient between .1 and 10

	  for (int z=denominator_ptr->get_min_index();z<= denominator_ptr->get_max_index();z++)
	    for (int y=(*denominator_ptr)[z].get_min_index();y<= (*denominator_ptr)[z].get_max_index();y++)
	      for (int x=(*denominator_ptr)[z][y].get_min_index();x<= (*denominator_ptr)[z][y].get_max_index();x++)
	      {
		const float sensitivity_at_voxel = (*sensitivity_image_ptr)[z][y][x];
		float& denominator_at_voxel = (*denominator_ptr)[z][y][x];
		const float sum = 1 + denominator_at_voxel;
		// bound denominator between 1/10 and 1*10
		denominator_at_voxel =
		  sensitivity_at_voxel*max(min(sum, 10.F),1/10.F);
	      }
	}
	
      }		
	
      if (do_rim_truncation)
	divide_and_truncate(*multiplicative_update_image_ptr, 
			    *denominator_ptr, 
			    rim_truncation_image,
			    count);
      else
	divide_array(*multiplicative_update_image_ptr, 
		     *denominator_ptr);
    }
    
    cerr<<"Number of (cancelled) singularities in Sensitivity division: "
      <<count<<endl;
  }
  
  
  //MJ 05/03/2000 moved this inside the update function
  
  
  *multiplicative_update_image_ptr*= 
    static_cast<float>(get_parameters().num_subsets);
  
  
  if(get_parameters().inter_update_filter_interval>0 &&
     !is_null_ptr(get_parameters().inter_update_filter_ptr) &&
     !(subiteration_num%get_parameters().inter_update_filter_interval))
  {
    
    cerr<<endl<<"Applying inter-update filter"<<endl;
    get_parameters().inter_update_filter_ptr->apply(current_image_estimate); 
    
  }
  
  // KT 17/08/2000 limit update
  // TODO move below thresholding?
  if (get_parameters().write_update_image)
  {
    // allocate space for the filename assuming that
    // we never have more than 10^49 subiterations ...
    char * fname = new char[get_parameters().output_filename_prefix.size() + 60];
    sprintf(fname, "%s_update_%d", get_parameters().output_filename_prefix.c_str(), subiteration_num);
    
    // Write it to file
    get_parameters().output_file_format_ptr->
      write_to_file(fname, *multiplicative_update_image_ptr);
    delete fname;
  }
  
  if (subiteration_num != 1)
    {
      const float current_min =
	multiplicative_update_image_ptr->find_min();
      const float current_max = 
	multiplicative_update_image_ptr->find_max();
      const float new_min = 
	static_cast<float>(get_parameters().minimum_relative_change);
      const float new_max = 
	static_cast<float>(get_parameters().maximum_relative_change);
      cerr << "Update image old min,max: " 
	   << current_min
	   << ", " 
	   << current_max
	   << ", new min,max " 
	   << max(current_min, new_min) << ", " << min(current_max, new_max)
	   << endl;

      threshold_upper_lower(multiplicative_update_image_ptr->begin_all(),
			    multiplicative_update_image_ptr->end_all(), 
			    new_min, new_max);      
    }  

  current_image_estimate *= *multiplicative_update_image_ptr; 
  
  
#ifndef PARALLEL
  //cerr << "Subset : " << subset_timer.value() << "secs " <<endl;
#else // PARALLEL
  timerSubset.Stop();
  cerr << "Subset: " << timerSubset.GetTime() << "secs" << endl;
#endif
  
}


END_NAMESPACE_STIR
