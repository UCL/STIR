//
// $Id$
//
/*!

  \file
  \ingroup LogLikBased_buildblock
  
  \brief  implementation of the LogLikelihoodBasedReconstruction class 
    
  \author Matthew Jacobson
  \author Kris Thielemans
  \author Sanida Mustafovic
  \author PARAPET project
      
  $Date$
        
  $Revision$
*/

#include "LogLikBased/LogLikelihoodBasedReconstruction.h"
#include "LogLikBased/common.h"
// for checks between sensitivity and target_image
#include "DiscretisedDensityOnCartesianGrid.h"
// for set_projectors_and_symmetries
#include "recon_buildblock/distributable.h"

#include "Viewgram.h"
#include "recon_array_functions.h"
#include <iostream>
#include <typeinfo>

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
  
  IterativeReconstruction::recon_set_up(target_image_ptr);
  
  
  if(get_parameters().sensitivity_image_filename=="1")
  {
    sensitivity_image_ptr=target_image_ptr->get_empty_discretised_density();
    sensitivity_image_ptr->fill(1.0);  
  }
  else
  {       
    
    sensitivity_image_ptr = 
      DiscretisedDensity<3,float>::read_from_file(get_parameters().sensitivity_image_filename);   

    if (typeid(*sensitivity_image_ptr) != typeid(*target_image_ptr))
      error("sensitivity image and target_image should be the same type of DiscretisedDensity. Sorry.\n");
    if (sensitivity_image_ptr->get_origin() != target_image_ptr->get_origin())
      error("Currently, sensitivity and target_image should have the same origin. Sorry.\n");
    if (sensitivity_image_ptr->get_index_range() != target_image_ptr->get_index_range())
      error("Currently, sensitivity and target_image should have the same index ranges. Sorry.\n");
    {
      DiscretisedDensityOnCartesianGrid<3,float> const *sens_ptr =
	       dynamic_cast<DiscretisedDensityOnCartesianGrid<3,float> const *>(sensitivity_image_ptr.get());
      if (sens_ptr != 0)
      {
	// we can now check on grid_spacing
	DiscretisedDensityOnCartesianGrid<3,float> const *image_ptr =
	  dynamic_cast<DiscretisedDensityOnCartesianGrid<3,float> const *>(target_image_ptr.get());
	  if (sens_ptr->get_grid_spacing() != image_ptr->get_grid_spacing())
	    error("Currently, sensitivity and target_image should have the same grid spacing. Sorry.\n");
      }
    }
    // TODO ensure compatible info for any type of DiscretisedDensity
  }
  
  
  if (get_parameters().additive_projection_data_filename != "0")
  {
    additive_projection_data_ptr = 
      ProjData::read_from_file(get_parameters().additive_projection_data_filename);
  };
  

  // set projectors to be used for the calculations
  
  get_parameters().projector_pair_ptr->set_up(get_parameters().proj_data_ptr->get_proj_data_info_ptr()->clone(), 
                                              target_image_ptr);
  set_projectors_and_symmetries(get_parameters().projector_pair_ptr->get_forward_projector_sptr(), 
                                get_parameters().projector_pair_ptr->get_back_projector_sptr(), 
                                get_parameters().projector_pair_ptr->get_back_projector_sptr()->get_symmetries_used()->clone());
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


float LogLikelihoodBasedReconstruction::sum_projection_data() const
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
