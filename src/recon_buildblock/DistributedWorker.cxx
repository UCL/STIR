/*
    Copyright (C) 2007- 2012, Hammersmith Imanet Ltd
    Copyright (C) 2013-2014, University College London
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
  \ingroup distributable

  \brief Implementation of stir::DistributedWorker()

  \author Tobias Beisel  
  \author Kris Thielemans
  \author Alexey Zverovich (idea of using main() function)
*/
#include "stir/recon_buildblock/DistributedWorker.h"
#include "stir/recon_buildblock/distributed_functions.h"
#include "stir/recon_buildblock/distributed_test_functions.h"
#include "stir/ExamInfo.h"
#include "stir/Viewgram.h"
#include "stir/RelatedViewgrams.h"
#include "stir/DataSymmetriesForViewSegmentNumbers.h"

#include "stir/ProjDataInMemory.h"
#include "stir/DiscretisedDensity.h"
#include "stir/HighResWallClockTimer.h"
#include "stir/is_null_ptr.h"
#include "stir/Succeeded.h"
#include "stir/info.h"
#include "stir/warning.h"
#include "stir/error.h"
#include <boost/format.hpp>
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h" // needed for RPC functions
#include <exception>

#include "stir/recon_buildblock/distributable_main.h"

int main(int argc, char **argv)
{

  int return_value = EXIT_FAILURE;
  try
    {
#ifndef STIR_MPI 
      return stir::distributable_main(argc, argv);
#else

      //processor-id within parallel Communicator
      int my_rank;  
        
      //saves the name of a processor
      char processor_name[MPI_MAX_PROCESSOR_NAME];           
      //length of the processor-name
      int namelength;       
         
      MPI_Init(&argc, &argv) ; /*Initializes the start up for MPI*/
      MPI_Comm_rank(MPI_COMM_WORLD, &my_rank) ; /*Gets the rank of the Processor*/   
      MPI_Comm_size(MPI_COMM_WORLD, &distributed::num_processors) ; /*Finds the number of processes being used*/     
      MPI_Get_processor_name(processor_name, &namelength);
      
      stir::info(boost::format("Process %d of %d on %s") 
	   % my_rank % distributed::num_processors % processor_name);
         
      //master
      if (my_rank==0)
        {
	  if (distributed::num_processors<2)
	    {
	      stir::error("STIR MPI processing needs more than 1 processor");
	      return_value = EXIT_FAILURE;
	    }
	  else
	    {
	      return_value=stir::distributable_main(argc, argv);
	      if (distributed::total_rpc_time_slaves!=0) 
		stir::info(boost::format("Total time used for RPC-processing: %1%") %
		     distributed::total_rpc_time_slaves);
	    }
	}
      else //slaves
        {           
          //create Slave Object
          stir::DistributedWorker<stir::DiscretisedDensity<3,float> > worker;
                
          //start Slave Process:
          worker.start();
          return_value = EXIT_SUCCESS;
        }   
#endif

    }
  catch (std::string& error_string)
    {
      // don't print yet, as error() already does that at the moment
      // std::cerr << error_string << std::endl;
      return_value = EXIT_FAILURE;
    }
  catch (std::exception& e)
    {
      stir::warning(e.what());
      return_value = EXIT_FAILURE;
    }
#ifdef STIR_MPI
  MPI_Finalize();
#endif
  return return_value;
}

  
namespace stir 
{
        
  template <typename TargetT>
  DistributedWorker<TargetT>::DistributedWorker() 
  {
    this->set_defaults();
    MPI_Comm_rank(MPI_COMM_WORLD, &this->my_rank) ; /*Gets the rank of the Processor*/   
  }
  
  template <typename TargetT>
  DistributedWorker<TargetT>::~DistributedWorker() 
  {
  }

  template <typename TargetT>
  void DistributedWorker<TargetT>::set_defaults()
  {
    log_likelihood_ptr=NULL;
    zero_seg0_end_planes=false;
    cache_enabled=false;
  }   
   
  template <typename TargetT>
  void DistributedWorker<TargetT>::start()
  {
   
    // keep-on waiting for new tasks
    while (true)
      {
        //Receive task broadcasted by Master
        const int task_id = distributed::receive_int_value(-1);
   
        switch(task_id)
          {
          case task_stop_processing: // done with processing
            {
              return;
            }

          case task_setup_distributable_computation:
            {
              this->setup_distributable_computation();
              break;
            } 

          case task_do_distributable_gradient_computation:
            {
              this->distributable_computation(RPC_process_related_viewgrams_gradient);
              break;
            }
	  case task_do_distributable_loglikelihood_computation:
	    {
	      this->distributable_computation(RPC_process_related_viewgrams_accumulate_loglikelihood);
	      break;
	    }
          case task_do_distributable_sensitivity_computation:
            {
              this->distributable_computation(RPC_process_related_viewgrams_sensitivity_computation);
              break;
            }

      /*
	case task_do_distributable_sensitivity_computation;break;
      */
          default:
            {
              error("Internal error: Slave %d received unknown task-id %d", this->my_rank, task_id);
            } 
          } // end switch task_id
      } // infinite loop
  }
   
  template <typename TargetT>
  void DistributedWorker<TargetT>::setup_distributable_computation()
  {
    //Receive zero_seg_end_planes
    this->zero_seg0_end_planes = distributed::receive_bool_value(-1,-1);
                                        
    //receive target image pointer
    distributed::receive_and_set_image_parameters(this->target_sptr, image_buffer_size, -1, 0);
                        
    //Receive input_image values
    MPI_Status status;
    status = distributed::receive_image_values_and_fill_image_ptr(this->target_sptr, this->image_buffer_size, 0);          
    //construct exam_info_ptr and projection_data_info_ptr
    distributed::receive_and_construct_exam_and_proj_data_info_ptr(this->exam_info_sptr, this->proj_data_info_sptr, 0);
                   
    //initialize projectors and call set_up
    distributed::receive_and_initialize_projectors(this->proj_pair_sptr, 0);
                
    proj_pair_sptr->set_up(this->proj_data_info_sptr, this->target_sptr);
                
    //some values to configure tests
    int configurations[4];
    status=distributed::receive_int_values(configurations, 4, distributed::STIR_MPI_CONF_TAG);
        
    status=distributed::receive_double_values(&distributed::min_threshold, 1, distributed::STIR_MPI_CONF_TAG);
                                        
    (configurations[0]==1)?distributed::test=true:distributed::test=false;
    (configurations[1]==1)?distributed::test_send_receive_times=true:distributed::test_send_receive_times=false;
    (configurations[2]==1)?distributed::rpc_time=true:distributed::rpc_time=false;
    (configurations[3]==1)?cache_enabled=true:cache_enabled=false;
                        
#ifndef NDEBUG
    if (distributed::test && my_rank==1) distributed::test_parameter_info_slave(proj_pair_sptr->stir::ParsingObject::parameter_info());
#endif

#if 0
    //create new Object of objective_function (but no longer used, so commented out)
    shared_ptr<PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT > > objective_function_ptr =
      new PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT >();
                       
    //set values in objective_function
    objective_function_ptr->set_zero_seg0_end_planes(zero_seg0_end_planes);
    objective_function_ptr->set_projector_pair_sptr(proj_pair_sptr);
#endif
                        
    // reset cache-stores, they will be initialised if we need them
    this->proj_data_ptr.reset();
    this->binwise_correction.reset(); 
    this->mult_proj_data_sptr.reset(); 
  } // set_up

  template <typename TargetT>
  void DistributedWorker<TargetT>::
  distributable_computation(RPC_process_related_viewgrams_type * RPC_process_related_viewgrams)
  {     
    shared_ptr<TargetT> input_image_ptr = this->target_sptr; // use the target_sptr member as we don't need its values anyway
                
    shared_ptr<DataSymmetriesForViewSegmentNumbers> 
      symmetries_sptr(this->proj_pair_sptr->get_symmetries_used()->clone());
                
#ifndef NDEBUG
    if (distributed::test && my_rank==1) 
      {
        distributed::test_image_estimate_slave();
        distributed::test_parameter_info_slave(proj_data_info_sptr->parameter_info());
        distributed::test_bool_value_slave();
        distributed::test_int_value_slave();
        distributed::test_int_values_slave();
        distributed::test_viewgram_slave(proj_data_info_sptr);
      }
#endif    
                 
    HighResWallClockTimer t;
                
      {
	if (distributed::receive_bool_value(USE_DOUBLE_ARG_TAG,-1))
	  {
	    this->log_likelihood_ptr = new double;
	    *this->log_likelihood_ptr=0.0;
	  }
	else
	  {
	    this->log_likelihood_ptr = 0;
	  }
                
        //Receive input_image values 
        MPI_Status status = distributed::receive_image_values_and_fill_image_ptr(input_image_ptr, this->image_buffer_size, 0);

	shared_ptr<TargetT> output_image_ptr;
	if (distributed::receive_bool_value(USE_OUTPUT_IMAGE_ARG_TAG,-1))
	  {
	    output_image_ptr.reset(this->target_sptr->get_empty_copy());
	    // already set to zero
	    //output_image_ptr->fill(0.F);
	  }
                   
        //loop to receive viewgrams until received END_ITERATION_TAG
        while (true)
          {
            // TODO, get rid of pointers somehow
            RelatedViewgrams<float>* viewgrams = NULL;
            RelatedViewgrams<float>* additive_binwise_correction_viewgrams = NULL;
            RelatedViewgrams<float>* mult_viewgrams_ptr = NULL;
            int count=0, count2=0;
                                
            //receive vs_num values     
            ViewSegmentNumbers vs;
            status = distributed::receive_view_segment_numbers(vs, MPI_ANY_TAG);
                        
            /*check whether to
             *  - use a viewgram already received in previous iteration
             *  - receive a new viewgram
             *  - end the iteration
             */         
            if (status.MPI_TAG==REUSE_VIEWGRAM_TAG) //use a viewgram already available
              {                        
                viewgrams = new RelatedViewgrams<float>(proj_data_ptr->get_related_viewgrams(vs, symmetries_sptr));
                if (!is_null_ptr(binwise_correction))
                  additive_binwise_correction_viewgrams = 
                    new RelatedViewgrams<float>(binwise_correction->get_related_viewgrams(vs, symmetries_sptr));
                if (!is_null_ptr(mult_proj_data_sptr))
                  mult_viewgrams_ptr = 
                    new RelatedViewgrams<float>(mult_proj_data_sptr->get_related_viewgrams(vs, symmetries_sptr));
              } 
            else if (status.MPI_TAG==NEW_VIEWGRAM_TAG) //receive a message with a new viewgram
              {
#ifndef NDEBUG
                //run test for related viewgrams
                if (distributed::test && my_rank==1 && distributed::first_iteration==true) distributed::test_related_viewgrams_slave(proj_data_info_sptr, symmetries_sptr);
#endif          
                //receive info if additive_binwise_correction_viewgrams are NULL        
                const bool add_bin_corr_viewgrams=distributed::receive_bool_value(BINWISE_CORRECTION_TAG, 0);
                if (add_bin_corr_viewgrams) 
                  {
                    distributed::receive_and_construct_related_viewgrams(additive_binwise_correction_viewgrams, proj_data_info_sptr, symmetries_sptr, 0);
                  }

                //receive info if mult_viewgrams_ptr are NULL   
                const bool mult_viewgrams=distributed::receive_bool_value(BINWISE_MULT_TAG, 0);         
                if (mult_viewgrams) 
                  {
                    distributed::receive_and_construct_related_viewgrams(mult_viewgrams_ptr, proj_data_info_sptr, symmetries_sptr, 0);
                  }

                // measured viewgrams
                distributed::receive_and_construct_related_viewgrams(viewgrams, proj_data_info_sptr, symmetries_sptr, 0); 
                        
                //save Viewgrams to ProjDataInMemory object
                if(cache_enabled)
                  {
                    if (is_null_ptr(this->proj_data_ptr))
                      this->proj_data_ptr.reset(new ProjDataInMemory(this->exam_info_sptr,this->proj_data_info_sptr, /*init_with_0*/ false));

                    if (proj_data_ptr->set_related_viewgrams(*viewgrams)==Succeeded::no)
                      error("Slave %i: Storing viewgrams failed!\n", my_rank);
                                
                    if (add_bin_corr_viewgrams) 
                      {
                        if (is_null_ptr(binwise_correction))
                          binwise_correction.reset(new ProjDataInMemory(this->exam_info_sptr,this->proj_data_info_sptr, /*init_with_0*/ false));
                        
                        if (binwise_correction->set_related_viewgrams(*additive_binwise_correction_viewgrams)==Succeeded::no)
                          error("Slave %i: Storing additive_binwise_correction_viewgrams failed!\n", my_rank);
                      }

                    if (mult_viewgrams) 
                      {
                        if (is_null_ptr(mult_proj_data_sptr))
                          mult_proj_data_sptr.reset(new ProjDataInMemory(this->exam_info_sptr,this->proj_data_info_sptr, /*init_with_0*/ false));

                        if (mult_proj_data_sptr->set_related_viewgrams(*mult_viewgrams_ptr)==Succeeded::no)
                          error("Slave %i: Storing mult_viewgrams_ptr failed!\n", my_rank);
                      }
                  }
              }
            else if (status.MPI_TAG==END_ITERATION_TAG)  //the iteration is completed --> send results
              { 
                //make reduction over computed output_images
                distributed::first_iteration=false;
		if (!is_null_ptr(output_image_ptr))
		  distributed::reduce_output_image(output_image_ptr, image_buffer_size, my_rank, 0);
		// and log_likelihood
		if (!is_null_ptr(log_likelihood_ptr))
		  {
		    double buffer = 0.0;
		    MPI_Reduce(log_likelihood_ptr, &buffer, /*size*/1, MPI_DOUBLE, MPI_SUM, /*destination*/ 0, MPI_COMM_WORLD);
		    delete log_likelihood_ptr;
		  }
                                
                if(distributed::rpc_time)
                  {
                    double send = distributed::total_rpc_time;
                    double receive;
                    MPI_Reduce(&send, &receive, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                    distributed::total_rpc_time = 0.0;
                  }
		// get out of infinite while loop
                break;
              }
            else 
              error("Slave received unknown tag");
            
            //measure time used for parallelized part
            if (distributed::rpc_time) {t.reset(); t.start();}
                        
            //call the actual calculation
            RPC_process_related_viewgrams(
                                          this->proj_pair_sptr->get_forward_projector_sptr(),
                                          this->proj_pair_sptr->get_back_projector_sptr(),
                                          output_image_ptr.get(), input_image_ptr.get(), viewgrams, 
                                          count, count2,
                                          log_likelihood_ptr,
                                          additive_binwise_correction_viewgrams,
                                          mult_viewgrams_ptr);
                                
            if (distributed::rpc_time)
              {
                t.stop();
                distributed::total_rpc_time=distributed::total_rpc_time + t.value();
                distributed::total_rpc_time_2=distributed::total_rpc_time_2 + t.value();
              } 

            int int_values[2];
            int_values[0]=count;
            int_values[1]=count2;
            //send count,count2 and ask for new work
            distributed::send_int_values(int_values, 2, AVAILABLE_NOTIFICATION_TAG, 0);
                        
            if (viewgrams!=NULL) delete viewgrams;
            if (additive_binwise_correction_viewgrams!=NULL) delete additive_binwise_correction_viewgrams;
            if (mult_viewgrams_ptr!=NULL) delete mult_viewgrams_ptr;
          }
        if (distributed::rpc_time)
	  stir::info(boost::format("Slave %1% used %2% seconds for PRC-processing.")
		     % my_rank % distributed::total_rpc_time_2);
      }         
  }

  // instantiation
  template class DistributedWorker<DiscretisedDensity<3,float> >;  
   
} // namespace stir
