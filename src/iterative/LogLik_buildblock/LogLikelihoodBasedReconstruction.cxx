//
// $Id$
//

#include "recon_buildblock/LogLikelihoodBasedReconstruction.h"

//
//
//---------------LogLikelihoodBasedReconstruction definitions-----------------
//
//


// LogLikelihoodBasedReconstruction::LogLikelihoodBasedReconstruction(char* parameter_filename)
//  :IterativeReconstruction(parameter_filename)
void LogLikelihoodBasedReconstruction::LogLikelihoodBasedReconstruction_ctor(char* parameter_filename)
{

  IterativeReconstruction_ctor(parameter_filename);

  sensitivity_image_ptr=NULL;
  additive_projection_data_ptr = NULL;


}

//LogLikelihoodBasedReconstruction::~LogLikelihoodBasedReconstruction()
void LogLikelihoodBasedReconstruction::LogLikelihoodBasedReconstruction_dtor()
{

   delete sensitivity_image_ptr;
   delete additive_projection_data_ptr;

  IterativeReconstruction_dtor();

}


//void LogLikelihoodBasedReconstruction::recon_set_up(PETImageOfVolume &target_image)
void LogLikelihoodBasedReconstruction::loglikelihood_common_recon_set_up(PETImageOfVolume &target_image)
{

  //IterativeReconstruction::recon_set_up(target_image);
   iterative_common_recon_set_up(target_image);


   sensitivity_image_ptr=new PETImageOfVolume(target_image.get_empty_copy());

   if(get_parameters().sensitivity_image_filename=="1")
     sensitivity_image_ptr->fill(1.0);
  
   else
     {
       // MJ 05/03/2000 replaced by interfile
       // TODO ensure compatable sizes of initial image and sensitivity

       *sensitivity_image_ptr = read_interfile_image(get_parameters().sensitivity_image_filename.c_str());   
     }
   /*
   //MJ 10/04/2000 added zoom
   zoom_image(*sensitivity_image_ptr,get_parameters().zoom,
	      get_parameters().Xoffset,
	      get_parameters().Yoffset,
	      get_parameters().output_image_size);

	      */

  if (get_parameters().additive_projection_data_filename != "0")
    {
      additive_projection_data_ptr = new PETSinogramOfVolume(read_interfile_PSOV(get_parameters().additive_projection_data_filename.c_str()));
    };


}




//void LogLikelihoodBasedReconstruction::end_of_iteration_processing(PETImageOfVolume &current_image_estimate)
void LogLikelihoodBasedReconstruction::loglikelihood_common_end_of_iteration_processing(PETImageOfVolume &current_image_estimate)
{

  //IterativeReconstruction::end_of_iteration_processing(current_image_estimate);
  iterative_common_end_of_iteration_processing(current_image_estimate);

    // Save intermediate (or last) iteration      
  if((!(subiteration_num%get_parameters().save_interval)) || subiteration_num==get_parameters().num_subiterations ) 
    {
      
    // KTxxx use output_filename_prefix
      char fname[max_filename_length];
      sprintf(fname, "%s_%d", get_parameters().output_filename_prefix.c_str(), subiteration_num);
	       
      if(get_parameters().do_post_filtering && subiteration_num==get_parameters().num_subiterations)
	{
	  cerr<<endl<<"Applying post-filter"<<endl;
	  get_parameters().post_filter.apply(current_image_estimate);

	  cerr << "  min and max after post-filtering " << current_image_estimate.find_min() 
	       << " " << current_image_estimate.find_max() << endl <<endl;
	}
 
     // Write it to file
      write_basic_interfile(fname, current_image_estimate);
 
    }


}




//MJ 03/01/2000 computes the negative of the loglikelihood function (minimization).
void LogLikelihoodBasedReconstruction::compute_loglikelihood(float* accum,
						       const PETImageOfVolume& current_image_estimate,
						       const PETImageOfVolume& sensitivity_image,
						       const PETSinogramOfVolume* proj_dat,
						       const int magic_number)
{



  *accum=0.0;

  

  distributable_accumulate_loglikelihood(current_image_estimate,proj_dat,
					 1, 1,
					 get_parameters().proj_data_ptr->get_num_views()/4,
					 -get_parameters().max_segment_num_to_process, 
					 get_parameters().max_segment_num_to_process, 
					 get_parameters().zero_seg0_end_planes, accum,
					 additive_projection_data_ptr);

 *accum/=magic_number;
 PETImageOfVolume temp_image=sensitivity_image;
 temp_image*=current_image_estimate;
 *accum+=temp_image.sum()/get_parameters().num_subsets; 
 cerr<<endl<<"Image Energy="<<temp_image.sum()/get_parameters().num_subsets<<endl;

}


float LogLikelihoodBasedReconstruction::sum_projection_data()
{

	float counts=0.0;
       
	for (int segment_num = -get_parameters().max_segment_num_to_process; segment_num <= get_parameters().max_segment_num_to_process; segment_num++)
	  {
	
	  //first adjust sinograms
	  PETSegmentByView  sino=get_parameters().proj_data_ptr->get_segment_view_copy(segment_num);
	 
	  if(segment_num==0 && get_parameters().zero_seg0_end_planes)
	    {
	      sino[sino.get_min_ring()].fill(0.0);
	      sino[sino.get_max_ring()].fill(0.0);
	    } 
   
	  truncate_rim(sino,rim_truncation_sino);
	
	  //now take totals
	  counts+=sino.sum();
	   
	}
 
	return counts;

}
