//
// $Id$
//

/*!

  \file
  
  \brief  implementation of the OSMAPOSLReconstruction class 
  \ingroup OSMAPOSL
    
  \author Matthew Jacobson
  \author Kris Thielemans
  \author PARAPET project
      
  \date $Date$
        
  \version $Revision$
  
*/

#include "OSMAPOSL/OSMAPOSLReconstruction.h"
#include "recon_array_functions.h"
#include "DiscretisedDensity.h"
#include "LogLikBased/common.h"
#include <memory>
#include <iostream>

#ifndef TOMO_NO_NAMESPACES
using std::auto_ptr;
using std::cerr;
using std::ends;
using std::endl;
#endif

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

  // TODO adapt for priors

  // TODO dangerous for out-of-range, but 'old-style' ostrstream seems to need this

  char str[10000];
  ostrstream s(str, 10000);

  // TODO add prior
  if(parameters.inter_update_filter_interval>0) s<<"IMF-";
  if(parameters.num_subsets>1) s<<"OS";
  s<<"EM";
  if(parameters.inter_iteration_filter_interval>0) s<<"S";
  s<<ends;

  return s.str();

}

void OSMAPOSLReconstruction::recon_set_up(shared_ptr <DiscretisedDensity<3,float> > const& target_image_ptr)
{
  LogLikelihoodBasedReconstruction::recon_set_up(target_image_ptr);

  if(parameters.enforce_initial_positivity) 
    truncate_min_to_small_positive_value(*target_image_ptr);

  if(parameters.inter_update_filter_interval>0 && !parameters.inter_update_filter.kernels_built)
    {
      cerr<<endl<<"Building inter-update filter kernel"<<endl;

      parameters.inter_update_filter.build(
                       *target_image_ptr,
		       parameters.inter_update_filter_fwhmxy_dir,
		       parameters.inter_update_filter_fwhmz_dir,
		       (float) parameters.inter_update_filter_Nxy_dir,
		       (float) parameters.inter_update_filter_Nz_dir);
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


  distributable_compute_gradient(*multiplicative_update_image_ptr, 
                                 current_image_estimate, 
                                 parameters.proj_data_ptr, 
                                 subset_num, 
                                 parameters.num_subsets, 
                                 0, 
                                 parameters.max_segment_num_to_process, 
                                 parameters.zero_seg0_end_planes, 
                                 NULL, 
                                 additive_projection_data_ptr);



  // divide by sensitivity

  {
    // KT 05/11/98 clean now
    int count = 0;
    if (parameters.MAP_model == "")
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

      compute_prior_gradient(*denominator_ptr, current_image_estimate); 

      *denominator_ptr += *sensitivity_image_ptr;

      //MJ 08/08/99 added negative truncation
      truncate_min_to_small_positive_value(*denominator_ptr);

     
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
  

  if(parameters.inter_update_filter_interval>0 && !(subiteration_num%parameters.inter_update_filter_interval))
    {

      cerr<<endl<<"Applying inter-update filter"<<endl;
      parameters.inter_update_filter.apply(current_image_estimate); 

    }

  
  current_image_estimate *= *multiplicative_update_image_ptr; 
  


#ifndef PARALLEL
  //cerr << "Subset : " << subset_timer.value() << "secs " <<endl;
#else // PARALLEL
  timerSubset.Stop();
  cerr << "Subset: " << timerSubset.GetTime() << "secs" << endl;
#endif

}


END_NAMESPACE_TOMO
