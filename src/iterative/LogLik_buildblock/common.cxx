//
// $Id$: $Date$
//
/*!

  \file
  \ingroup LogLikBased_buildblock

  \brief Implementation of functions defined in LogLikBased/common.h and
  LogLikBased/distributable.h

  \author Kris Thielemans (based on earlier work by Alexey Zverovich and Matthew Jacobson)
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/
#include "shared_ptr.h"
#include "LogLikBased/common.h"
#include "recon_array_functions.h"
#include "ArrayFunction.h"
#include "RelatedViewgrams.h"
#include "DiscretisedDensity.h"
#include "ProjDataFromStream.h"
#include "new_recon_buildblock/ForwardProjectorByBin.h"
#include "new_recon_buildblock/BackProjectorByBin.h"
#include "recon_buildblock/distributable.h"

START_NAMESPACE_TOMO


//! Call-back function for compute_gradient
static RPC_process_related_viewgrams_type RPC_process_related_viewgrams_gradient;

//! Call-back function for accumulate_loglikelihood
static RPC_process_related_viewgrams_type RPC_process_related_viewgrams_accumulate_loglikelihood;

//! Call-back function for compute_sensitivity_image
static RPC_process_related_viewgrams_type RPC_process_related_viewgrams_sensitivity;

void distributable_compute_gradient(DiscretisedDensity<3,float>& output_image,
				    const DiscretisedDensity<3,float>& input_image,
				    const shared_ptr<ProjData>& proj_dat,
				    int subset_num, int num_subsets,
				    int min_segment, int max_segment,
				    bool zero_seg0_end_planes,
				    float* log_likelihood_ptr,
				    shared_ptr<ProjData> const& additive_binwise_correction)
{
  distributable_computation(&output_image, &input_image,
                            proj_dat, true, //i.e. do read projection data
                            subset_num, num_subsets,
                            min_segment, max_segment,
                            zero_seg0_end_planes,
                            log_likelihood_ptr,
                            additive_binwise_correction,
                            &RPC_process_related_viewgrams_gradient);
}


void distributable_accumulate_loglikelihood(
				    const DiscretisedDensity<3,float>& input_image,
				    const shared_ptr<ProjData>& proj_dat,
				    int subset_num, int num_subsets,
				    int min_segment, int max_segment,
				    bool zero_seg0_end_planes,
				    float* log_likelihood_ptr,
				    shared_ptr<ProjData> const& additive_binwise_correction)
{
  distributable_computation(NULL, &input_image, 
                            proj_dat, true, //i.e. do read projection data
                            subset_num, num_subsets,
                            min_segment, max_segment,
                            zero_seg0_end_planes,
                            log_likelihood_ptr,
                            additive_binwise_correction,
                            &RPC_process_related_viewgrams_accumulate_loglikelihood);
}


void distributable_compute_sensitivity_image(DiscretisedDensity<3,float>& result,
					     shared_ptr<ProjDataInfo> const& proj_data_info_ptr,
					     DiscretisedDensity<3,float> const* attenuation_image_ptr,
					     const int subset_num,
					     const int num_subsets,
					     const int min_segment,
					     const int max_segment,
					     bool zero_seg0_end_planes,
					     shared_ptr<ProjData> const& multiplicative_sinogram_ptr /* = NULL */)
{


  // create an empty ProjData, as this is needed by distributable_computation
  // it will never read the data though

  shared_ptr<ProjData> proj_data_ptr =
    new ProjDataFromStream(proj_data_info_ptr, static_cast<iostream *>(NULL));

  distributable_computation(&result,
                            attenuation_image_ptr,
                            proj_data_ptr,
                            false, //i.e. do not read projection data
                            subset_num,
                            num_subsets,
                            min_segment,
                            max_segment,
                            zero_seg0_end_planes,
                            NULL,
                            multiplicative_sinogram_ptr,
                            RPC_process_related_viewgrams_sensitivity);
  
};

//////////// RPC functions


void RPC_process_related_viewgrams_gradient(DiscretisedDensity<3,float>* output_image_ptr, 
                                            const DiscretisedDensity<3,float>* input_image_ptr, 
                                            RelatedViewgrams<float>* measured_viewgrams_ptr,
                                            int& count, int& count2, float* log_likelihood_ptr /* = NULL */,
                                            const RelatedViewgrams<float>* additive_binwise_correction_ptr)
{

  assert(output_image_ptr != NULL);
  assert(input_image_ptr != NULL);
  assert(measured_viewgrams_ptr != NULL);

  RelatedViewgrams<float> estimated_viewgrams = measured_viewgrams_ptr->get_empty_copy();

  forward_projector_ptr->forward_project(estimated_viewgrams, *input_image_ptr);

  if (additive_binwise_correction_ptr != NULL)
  {
    // TODO
    //estimated_viewgrams += (*additive_binwise_correction_ptr);

    RelatedViewgrams<float>::iterator est_viewgrams_iter = 
      estimated_viewgrams.begin();
    RelatedViewgrams<float>::const_iterator add_cor_viewgrams_iter = 
      additive_binwise_correction_ptr->begin();
    for (;
         est_viewgrams_iter != estimated_viewgrams.end();
         ++est_viewgrams_iter, ++add_cor_viewgrams_iter)
      (*est_viewgrams_iter) += (*add_cor_viewgrams_iter);
  }

  // for sinogram division
      
  divide_and_truncate(*measured_viewgrams_ptr, estimated_viewgrams, rim_truncation_sino, count, count2, log_likelihood_ptr);
      
  back_projector_ptr->back_project(*output_image_ptr, *measured_viewgrams_ptr);
};      


void RPC_process_related_viewgrams_accumulate_loglikelihood(DiscretisedDensity<3,float>* output_image_ptr, 
                                                            const DiscretisedDensity<3,float>* input_image_ptr, 
                                                            RelatedViewgrams<float>* measured_viewgrams_ptr,
                                                            int& count, int& count2, float* log_likelihood_ptr,
                                                            const RelatedViewgrams<float>* additive_binwise_correction_ptr)
{

  assert(output_image_ptr == NULL);
  assert(input_image_ptr != NULL);
  assert(measured_viewgrams_ptr != NULL);
  assert(log_likelihood_ptr != NULL);

  RelatedViewgrams<float> estimated_viewgrams = measured_viewgrams_ptr->get_empty_copy();

  forward_projector_ptr->forward_project(estimated_viewgrams, *input_image_ptr);

  if (additive_binwise_correction_ptr != NULL)
  {
    // TODO
    //estimated_viewgrams += (*additive_binwise_correction_ptr);
    RelatedViewgrams<float>::iterator est_viewgrams_iter = 
		estimated_viewgrams.begin();
	RelatedViewgrams<float>::const_iterator add_cor_viewgrams_iter = 
		additive_binwise_correction_ptr->begin();
    for (;
         est_viewgrams_iter != estimated_viewgrams.end();
         ++est_viewgrams_iter, ++add_cor_viewgrams_iter)
    {
      (*est_viewgrams_iter) += (*add_cor_viewgrams_iter);
    }
  };
  
  RelatedViewgrams<float>::iterator meas_viewgrams_iter = 
	  measured_viewgrams_ptr->begin();
  RelatedViewgrams<float>::iterator est_viewgrams_iter = 
	  estimated_viewgrams.begin();
  for (;
       meas_viewgrams_iter != measured_viewgrams_ptr->end();
       ++meas_viewgrams_iter, ++est_viewgrams_iter)
    accumulate_loglikelihood(*meas_viewgrams_iter, 
                             *est_viewgrams_iter, 
                             rim_truncation_sino, log_likelihood_ptr);
};      


void RPC_process_related_viewgrams_sensitivity
(DiscretisedDensity<3,float>* output_image_ptr, 
 const DiscretisedDensity<3,float>* attenuation_image_ptr, 
 RelatedViewgrams<float>* viewgrams_ptr,
 int& count, int& count2, float* log_likelihood_ptr,
 const RelatedViewgrams<float>* multiplicative_binwise_correction_ptr)
{
  
  assert(output_image_ptr != NULL);
  assert(viewgrams_ptr != NULL);
  
  if (attenuation_image_ptr == NULL)
  {
    
    if (multiplicative_binwise_correction_ptr != NULL)
    {
      *viewgrams_ptr = *multiplicative_binwise_correction_ptr;
    }
    else
    {
      for (RelatedViewgrams<float>::iterator iter = viewgrams_ptr->begin(); 
      iter != viewgrams_ptr->end();
      ++iter)
        iter->fill(1.F);
    }
  }
  else
  {
    // do attenuation
    
    forward_projector_ptr->forward_project(*viewgrams_ptr, *attenuation_image_ptr);
    
    RelatedViewgrams<float>::iterator viewgrams_iter = 
      viewgrams_ptr->begin();
    for (; 
    viewgrams_iter != viewgrams_ptr->end();
    ++viewgrams_iter)
    {
      Viewgram<float>& viewgram = *viewgrams_iter;
      viewgram *= -1;
      in_place_exp(viewgram);
      truncate_rim(viewgram, rim_truncation_sino);
    }
    if (multiplicative_binwise_correction_ptr != NULL)
    {
      RelatedViewgrams<float>::iterator viewgrams_iter = 
        viewgrams_ptr->begin();
      RelatedViewgrams<float>::const_iterator mult_viewgrams_iter = 
        multiplicative_binwise_correction_ptr->begin();
      for (; 
           viewgrams_iter != viewgrams_ptr->end();
           ++viewgrams_iter, ++mult_viewgrams_iter)
      {
        *viewgrams_iter *= *mult_viewgrams_iter;
      }
    }
  }  

  if (RPC_slave_sens_zero_seg0_end_planes)
  {
    const int min_ax_pos_num = viewgrams_ptr->get_min_axial_pos_num();
    const int max_ax_pos_num = viewgrams_ptr->get_max_axial_pos_num();
    for (RelatedViewgrams<float>::iterator r_viewgrams_iter = viewgrams_ptr->begin();
         r_viewgrams_iter != viewgrams_ptr->end();
         ++r_viewgrams_iter)
    {
      if (r_viewgrams_iter->get_segment_num() == 0)
      {
        (*r_viewgrams_iter)[min_ax_pos_num].fill(0);
        (*r_viewgrams_iter)[max_ax_pos_num].fill(0);
      }
    }
  }  
  // TODO replace zeroing with calling back_project with min_ax, max_ax args
  back_projector_ptr->back_project(*output_image_ptr, *viewgrams_ptr);
}


END_NAMESPACE_TOMO
