//
// $Id$
//

/*!
  \file
  \ingroup OSMAPOSL  
  \brief  implementation of the OSMAPOSLReconstruction class 
    
  \author Matthew Jacobson
  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project
      
  \date $Date$
  \version $Revision$
*/

#include "OSMAPOSL/OSMAPOSLReconstruction.h"
#include "recon_array_functions.h"
#include "DiscretisedDensity.h"
#include "LogLikBased/common.h"
#include "tomo/TruncateMinToSmallPositiveValueImageProcessor.h"
#include "tomo/ChainedImageProcessor.h"
#include <memory>
#include <iostream>

#ifndef TOMO_NO_NAMESPACES
using std::auto_ptr;
using std::cerr;
using std::ends;
using std::endl;
#endif

// KT 17/08/2000 limit update
#include "NumericInfo.h"
// for write_update_image
#include "interfile.h"

START_NAMESPACE_TOMO

OSMAPOSLReconstruction::
OSMAPOSLReconstruction(const OSMAPOSLParameters& parameters_v)
: parameters(parameters_v)
{
  cerr<<parameters.parameter_info();
}


OSMAPOSLReconstruction::
OSMAPOSLReconstruction(const string& parameter_filename)
: parameters(parameter_filename)
{  

  cerr<<parameters.parameter_info();
}

string OSMAPOSLReconstruction::method_info() const
{

  // TODO add prior name?

  // TODO dangerous for out-of-range, but 'old-style' ostrstream seems to need this

  char str[10000];
  ostrstream s(str, 10000);


  if(parameters.inter_update_filter_interval>0) s<<"IMF-";
  if(parameters.num_subsets>1) s<<"OS";
  if (parameters.prior_ptr == 0 || 
      parameters.prior_ptr->get_penalisation_factor() == 0)
    s<<"EM";
  else
    s << "MAPOSL";
  if(parameters.inter_iteration_filter_interval>0) s<<"S";
  s<<ends;

  return s.str();

}

void OSMAPOSLReconstruction::recon_set_up(shared_ptr <DiscretisedDensity<3,float> > const& target_image_ptr)
{
  LogLikelihoodBasedReconstruction::recon_set_up(target_image_ptr);

  if(parameters.enforce_initial_positivity) 
    truncate_min_to_small_positive_value(*target_image_ptr);

  if(parameters.inter_update_filter_interval>0 && 
     parameters.inter_update_filter_ptr != 0)
    {
      cerr<<endl<<"Building inter-update filter kernel"<<endl;
      parameters.inter_update_filter_ptr->build_filter(*target_image_ptr);

      // ensure that the result image of the filter is positive
      parameters.inter_update_filter_ptr =
	new ChainedImageProcessor<3,float>(
				  parameters.inter_update_filter_ptr,
				  new  TruncateMinToSmallPositiveValueImageProcessor<float>
);

    }
  if (parameters.inter_iteration_filter_interval>0 && 
      parameters.inter_iteration_filter_ptr!=0)
    {
      // ensure that the result image of the filter is positive
      parameters.inter_iteration_filter_ptr =
	new ChainedImageProcessor<3,float>(
					   parameters.inter_iteration_filter_ptr,
					   new  TruncateMinToSmallPositiveValueImageProcessor<float>
);
    }
}




// KT 17/08/2000 limit update, new static function, will be moved somewhere else
static void
truncate_min_max(DiscretisedDensity<3,float>& image, 
                    const float new_min, const float new_max)
{
  float current_max = NumericInfo<float>().min_value();
  float current_min = NumericInfo<float>().max_value();

  for (int c1 = image.get_min_index(); c1 <= image.get_max_index(); ++c1)
    for (int c2 = image[c1].get_min_index(); c2 <= image[c1].get_max_index(); ++c2)
      for (int c3 = image[c1][c2].get_min_index(); c3 <= image[c1][c2].get_max_index(); ++c3)
      {
         const float current_value = image[c1][c2][c3];
         if (current_value > current_max)
           current_max = current_value;
         if (current_value < current_min)
           current_min = current_value;
         if (current_value > new_max)
           image[c1][c2][c3] = new_max;
         if (current_value < new_min)
           image[c1][c2][c3] = new_min;
      }

  cerr << "Update image old min,max: " 
       << current_min << ", " << current_max
       << ", new min,max " 
       << max(current_min, new_min) << ", " << min(current_max, new_max)
       << endl;
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
  
  if(parameters.randomise_subset_order && (subiteration_num-1)%parameters.num_subsets==0)
  {
    subset_array = randomly_permute_subset_order();
    
    cerr<<endl<<"Content ver.:"<<endl;
    
    for(int i=subset_array.get_min_index();i<=subset_array.get_max_index();i++) cerr<<subset_array[i]<<" ";
  };
  
  const int subset_num=parameters.randomise_subset_order ? subset_array[(subiteration_num-1)%parameters.num_subsets] : (subiteration_num+parameters.start_subset_num-1)%parameters.num_subsets;
  
  cerr<<endl<<"Now processing subset #: "<<subset_num<<endl;
  
  
  // KT 05/07/2000 made zero_seg0_end_planes int
  distributable_compute_gradient(*multiplicative_update_image_ptr, 
    current_image_estimate, 
    parameters.proj_data_ptr, 
    subset_num, 
    parameters.num_subsets, 
    0, 
    parameters.max_segment_num_to_process, 
    parameters.zero_seg0_end_planes!=0, 
    NULL, 
    additive_projection_data_ptr);
  
#endif  
  // divide by sensitivity  
  {
    // KT 05/11/98 clean now
    int count = 0;
    
    //std::cerr <<parameters.MAP_model << std::endl;
    
    if(parameters.prior_ptr == 0 || parameters.prior_ptr->get_penalisation_factor() == 0)     
    {
      divide_and_truncate(*multiplicative_update_image_ptr, 
	*sensitivity_image_ptr, 
	rim_truncation_image,
	count);
    }
    else
    {
      auto_ptr< DiscretisedDensity<3,float> > denominator_ptr = 
        auto_ptr< DiscretisedDensity<3,float> >(current_image_estimate.get_empty_discretised_density());
      
      
      parameters.prior_ptr->compute_gradient(*denominator_ptr, current_image_estimate); 
      
      if(parameters.MAP_model =="additive" )
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
	if(parameters.MAP_model =="multiplicative" )
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
	
      divide_and_truncate(*multiplicative_update_image_ptr,  
	*denominator_ptr, 
	rim_truncation_image,
	count);
    }
    
    cerr<<"Number of (cancelled) singularities in Sensitivity division: "
      <<count<<endl;
  }
  
  
  //MJ 05/03/2000 moved this inside the update function
  
  
  *multiplicative_update_image_ptr*= parameters.num_subsets;
  
  
  if(parameters.inter_update_filter_interval>0 &&
     parameters.inter_update_filter_ptr != 0 &&
     !(subiteration_num%parameters.inter_update_filter_interval))
  {
    
    cerr<<endl<<"Applying inter-update filter"<<endl;
    parameters.inter_update_filter_ptr->apply(current_image_estimate); 
    
  }
  
  // KT 17/08/2000 limit update
  // TODO move below thresholding?
  if (parameters.write_update_image)
  {
    // allocate space for the filename assuming that
    // we never have more than 10^49 subiterations ...
    char * fname = new char[parameters.output_filename_prefix.size() + 60];
    sprintf(fname, "%s_update_%d", get_parameters().output_filename_prefix.c_str(), subiteration_num);
    
    // Write it to file
    write_basic_interfile(fname, *multiplicative_update_image_ptr);
    delete fname;
  }
  
  if (subiteration_num != 1)
    truncate_min_max(*multiplicative_update_image_ptr, 
    static_cast<float>(parameters.minimum_relative_change), 
    static_cast<float>(parameters.maximum_relative_change));
  
  current_image_estimate *= *multiplicative_update_image_ptr; 
  
  
#ifndef PARALLEL
  //cerr << "Subset : " << subset_timer.value() << "secs " <<endl;
#else // PARALLEL
  timerSubset.Stop();
  cerr << "Subset: " << timerSubset.GetTime() << "secs" << endl;
#endif
  
}


END_NAMESPACE_TOMO
