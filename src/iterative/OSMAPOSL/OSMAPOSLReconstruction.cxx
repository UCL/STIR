//
// $Id$
//


#include "OSEM/OSMAPOSLReconstruction.h"

//
//
//---------------OSMAPOSLReconstruction definitions-----------------
//

string OSMAPOSLReconstruction::method_info() const
{

  // TODO dangerous for out-of-range, but 'old-style' ostrstream seems to need this

  char str[10000];
  ostrstream s(str, 10000);

  if(parameters.inter_update_filter_interval>0) s<<"IMF-";
  if(parameters.num_subsets>1) s<<"OS";
  s<<"EM";
  if(parameters.inter_iteration_filter_interval>0) s<<"S";
  s<<ends;

  return s.str();

}

void OSMAPOSLReconstruction::recon_set_up(PETImageOfVolume &target_image)
{
  //LogLikelihoodBasedReconstruction::recon_set_up(target_image);
  loglikelihood_common_recon_set_up(target_image);

  //MJ 05/03/2000 KT requested
  //TODO remove ZERO_TOL
  //Note: only strictly negative voxel values altered by this 
  if(parameters.enforce_initial_positivity) set_negatives_small(target_image);

  if(parameters.inter_update_filter_interval>0 && !parameters.inter_update_filter.kernels_built)
    {
      cerr<<endl<<"Building inter-update filter kernel"<<endl;

      parameters.inter_update_filter.build(
                       target_image,
		       parameters.inter_update_filter_fwhmxy_dir,
		       parameters.inter_update_filter_fwhmz_dir,
		       (float) parameters.inter_update_filter_Nxy_dir,
		       (float) parameters.inter_update_filter_Nz_dir);
    }

}





void OSMAPOSLReconstruction::update_image_estimate(PETImageOfVolume &current_image_estimate)
{


  //MJ 03/05/2000 necessaru until we have default constructors

  static const PETSinogramOfVolume& proj_dat = *parameters.proj_data_ptr;
  static const PETImageOfVolume &sensitivity_image = *sensitivity_image_ptr;

  // KT xxx removed view45 from parameters
  const int view45= proj_dat.scan_info.get_num_views()/4;

  // KT 05/11/98 use current_image_estimate sizes


 

#ifndef PARALLEL
  //CPUTimer subset_timer;
  //subset_timer.start();
#else // PARALLEL
  PTimer timerSubset;
  timerSubset.Start();
#endif // PARALLEL
  
  PETImageOfVolume multiplicative_update_image=current_image_estimate.get_empty_copy();

  
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


  distributable_compute_gradient(current_image_estimate, multiplicative_update_image, &proj_dat, subset_num, parameters.num_subsets, view45, 0, parameters.max_segment_num_to_process, parameters.zero_seg0_end_planes, NULL, additive_projection_data_ptr);



  // divide by sensitivity

  {
    // KT 05/11/98 clean now
    int count = 0;
    if (parameters.MAP_model == "")
    {

      divide_and_truncate(multiplicative_update_image, 
                          sensitivity_image, 
			  rim_truncation_image,
                          count);
    }
    else
    {
      PETImageOfVolume denominator = 
	compute_prior_gradient(current_image_estimate); 

      denominator += sensitivity_image;

      //MJ 08/08/99 added negative truncation
      set_negatives_small(denominator);

     
      divide_and_truncate(multiplicative_update_image,  
                          denominator, 
			  rim_truncation_image,
                          count);
    }
            
    cerr<<"Number of (cancelled) singularities in Sensitivity division: "
        <<count<<endl;
  }


  //MJ 05/03/2000 moved this inside the update function
  

  multiplicative_update_image*= parameters.num_subsets;
  

  //MJ 22/10/98 we were filtering multiplicative_update_image by mistake
  // KT 13/11/98 remove parameters. from subiteration_num 
  if(parameters.inter_update_filter_interval>0 && !(subiteration_num%parameters.inter_update_filter_interval))
    {

      cerr<<endl<<"Applying inter-update filter"<<endl;
      parameters.inter_update_filter.apply(current_image_estimate); //Do iterative filtering

    }

  
  current_image_estimate *= multiplicative_update_image; 
  


#ifndef PARALLEL
  //cerr << "Subset : " << subset_timer.value() << "secs " <<endl;
#else // PARALLEL
  timerSubset.Stop();
  cerr << "Subset: " << timerSubset.GetTime() << "secs" << endl;
#endif

}
