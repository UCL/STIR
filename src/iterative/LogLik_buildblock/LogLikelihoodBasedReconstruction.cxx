//
// $Id$: $Date$
//
/*!

  \file
  \ingroup LogLikBased_buildblock
  
  \brief  implementation of the LogLikelihoodBasedReconstruction class 
    
  \author Matthew Jacobson
  \author Kris Thielemans
  \author PARAPET project
      
  \date $Date$
        
  \version $Revision$
*/

#include "LogLikBased/LogLikelihoodBasedReconstruction.h"
#include "LogLikBased/common.h"
// for set_projectors_and_symmetries
#include "recon_buildblock/distributable.h"
#ifndef USE_PMRT
#include "recon_buildblock/ForwardProjectorByBinUsingRayTracing.h"
#include "recon_buildblock/BackProjectorByBinUsingInterpolation.h"
#else
#include "recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"
#include "recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#endif
#ifdef PROJSMOOTH
#include "recon_buildblock/PostsmoothingForwardProjectorByBin.h"
#endif

#include "interfile.h"
#include "Viewgram.h"
#include "recon_array_functions.h"
#include <iostream>

#ifndef TOMO_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif

START_NAMESPACE_TOMO


LogLikelihoodBasedReconstruction::LogLikelihoodBasedReconstruction()
{


  sensitivity_image_ptr=NULL;
  additive_projection_data_ptr = NULL;


}



void LogLikelihoodBasedReconstruction::recon_set_up(shared_ptr <DiscretisedDensity<3,float> > const& target_image_ptr)
{
  // TODO move to post_processing
  
  IterativeReconstruction::recon_set_up(target_image_ptr);
  
  sensitivity_image_ptr=target_image_ptr->get_empty_discretised_density();
  
  if(get_parameters().sensitivity_image_filename=="1")
    sensitivity_image_ptr->fill(1.0);
  
  else
  {       
    // TODO ensure compatible sizes of initial image and sensitivity
    
    sensitivity_image_ptr = 
      DiscretisedDensity<3,float>::read_from_file(get_parameters().sensitivity_image_filename);   
  }
  
  
  if (get_parameters().additive_projection_data_filename != "0")
  {
    additive_projection_data_ptr = 
      ProjData::read_from_file(get_parameters().additive_projection_data_filename);
  };
  

  // set projectors to be used for the calculations
  // TODO get type and parameters for projectors from *Parameters
#ifndef USE_PMRT
  shared_ptr<ForwardProjectorByBin> forward_projector_ptr =
    new ForwardProjectorByBinUsingRayTracing(get_parameters().proj_data_ptr->get_proj_data_info_ptr()->clone(), 
                                             target_image_ptr);
#else
  shared_ptr<ProjMatrixByBin> PM = 
    new  ProjMatrixByBinUsingRayTracing( target_image_ptr , get_parameters().proj_data_ptr->get_proj_data_info_ptr()->clone()); 	
  ForwardProjectorByBin* forward_projector_ptr =
    new ForwardProjectorByBinUsingProjMatrixByBin(PM); 
#endif
#ifdef PROJSMOOTH
  if (get_parameters().forward_proj_postsmooth_tang_kernel.get_length() > 1
      || get_parameters().forward_proj_postsmooth_ax_kernel.get_length() > 1)
    forward_projector_ptr =
      new PostsmoothingForwardProjectorByBin(forward_projector_ptr, 
					     get_parameters().forward_proj_postsmooth_tang_kernel,
					     get_parameters().forward_proj_postsmooth_ax_kernel,
					     get_parameters().forward_proj_postsmooth_smooth_segment_0_axially!=0);
#endif

#ifndef USE_PMRT
  shared_ptr<BackProjectorByBin> back_projector_ptr =
    new BackProjectorByBinUsingInterpolation(get_parameters().proj_data_ptr->get_proj_data_info_ptr()->clone(), 
                                             target_image_ptr);
#else
  BackProjectorByBin* back_projector_ptr =
    new BackProjectorByBinUsingProjMatrixByBin(PM); 
#endif
  set_projectors_and_symmetries(forward_projector_ptr, 
                                back_projector_ptr, 
                                back_projector_ptr->get_symmetries_used()->clone());
}




void LogLikelihoodBasedReconstruction::end_of_iteration_processing(DiscretisedDensity<3,float> &current_image_estimate)
{

  IterativeReconstruction::end_of_iteration_processing(current_image_estimate);

    // Save intermediate (or last) iteration      
  if((!(subiteration_num%get_parameters().save_interval)) || subiteration_num==get_parameters().num_subiterations ) 
    {      	       
      if(get_parameters().do_post_filtering && subiteration_num==get_parameters().num_subiterations)
	{
	  cerr<<endl<<"Applying post-filter"<<endl;
	  get_parameters().post_filter.apply(current_image_estimate);

	  cerr << "  min and max after post-filtering " << current_image_estimate.find_min() 
	       << " " << current_image_estimate.find_max() << endl <<endl;
	}
 
      // allocate space for the filename assuming that
      // we never have more than 10^49 subiterations ...
      char * fname = new char[get_parameters().output_filename_prefix.size() + 50];
      sprintf(fname, "%s_%d", get_parameters().output_filename_prefix.c_str(), subiteration_num);

     // Write it to file
      write_basic_interfile(fname, current_image_estimate);
      delete fname;
 
    }


}




//MJ 03/01/2000 computes the negative of the loglikelihood function (minimization).
float LogLikelihoodBasedReconstruction::compute_loglikelihood(
						       const DiscretisedDensity<3,float>& current_image_estimate,
						       const int magic_number)
{

  float accum=0.F;  

  // KT 25/05/2000 subset_num -> 0 (was 1)
  // KT 05/07/2000 made parameters.zero_seg0_end_planes int
  distributable_accumulate_loglikelihood(current_image_estimate,
                                         get_parameters().proj_data_ptr,
					 0,1,
					 -get_parameters().max_segment_num_to_process, 
					 get_parameters().max_segment_num_to_process, 
					 get_parameters().zero_seg0_end_planes != 0, &accum,
					 additive_projection_data_ptr);

  accum/=magic_number;
  auto_ptr<DiscretisedDensity<3,float> > temp_image_ptr = 
    auto_ptr<DiscretisedDensity<3,float> >(sensitivity_image_ptr->clone());
  *temp_image_ptr *=current_image_estimate;
  accum+=temp_image_ptr->sum()/get_parameters().num_subsets; 
  cerr<<endl<<"Image Energy="<<temp_image_ptr->sum()/get_parameters().num_subsets<<endl;
  
  return accum;
}


float LogLikelihoodBasedReconstruction::sum_projection_data()
{
  
  float counts=0.0F;
  
  for (int segment_num = -get_parameters().max_segment_num_to_process; segment_num <= get_parameters().max_segment_num_to_process; segment_num++)
  {
    for (int view_num = get_parameters().proj_data_ptr->get_min_view_num();
         view_num <= get_parameters().proj_data_ptr->get_max_view_num();
         ++view_num)
    {
      
      Viewgram<float>  viewgram=get_parameters().proj_data_ptr->get_viewgram(view_num,segment_num);
      
      //first adjust data
      
      // KT 05/07/2000 made parameters.zero_seg0_end_planes int
      if(segment_num==0 && get_parameters().zero_seg0_end_planes!=0)
      {
        viewgram[viewgram.get_min_axial_pos_num()].fill(0);
        viewgram[viewgram.get_max_axial_pos_num()].fill(0);
      } 
      
      truncate_rim(viewgram,rim_truncation_sino);
      
      //now take totals
      counts+=viewgram.sum();
    }
  }
  
  return counts;
  
}

END_NAMESPACE_TOMO
