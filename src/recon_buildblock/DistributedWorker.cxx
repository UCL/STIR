//
// $Id$
//
/*
    Copyright (C) 2007- $Date$, Hammersmith Imanet Ltd
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

  $Date$
*/
#include "stir/recon_buildblock/DistributedWorker.h"
#include "stir/recon_buildblock/distributed_functions.h"
#include "stir/recon_buildblock/distributed_test_functions.h"
#include "stir/HighResWallClockTimer.h"

  
namespace stir 
{
	
  template <typename TargetT>
  DistributedWorker<TargetT>::DistributedWorker() 
  {
    set_defaults();
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
  void DistributedWorker<TargetT>::start(int rank)
  {
    my_rank=rank;
   		
    //Receive objective function broadcasted by Master
    int obj_fct = distributed::receive_int_value(-1);
   
    if(obj_fct==200) //objective_function is PoissonLogLikelihoodWithLinearModelForMeanAndProjData
      {
	//Receive zero_seg_end_planes
	bool zero_seg0_end_planes = distributed::receive_bool_value(-1,-1);
					
	//receive target image pointer
	shared_ptr<TargetT > target_image_sptr;
	distributed::receive_and_set_image_parameters(target_image_sptr, image_buffer_size, -1, 0);
	   	   	
	//Receive input_image values
	MPI_Status status;
	status = distributed::receive_image_values_and_fill_image_ptr(target_image_sptr, image_buffer_size, 0);
	  
	//construct projection_data_info_ptr
	shared_ptr<ProjDataInfo> proj_data_info_sptr;
	distributed::receive_and_construct_proj_data_info_ptr(proj_data_info_sptr, 0);
		
	//initialize projectors and call set_up
	shared_ptr<ProjectorByBinPair> proj_pair_sptr;
	distributed::receive_and_initialize_projectors(proj_pair_sptr, 0);
		
	proj_pair_sptr->set_up(proj_data_info_sptr, target_image_sptr);
		
	//some values to configure tests
	int * configurations = new int[4];
	status=distributed::receive_int_values(configurations, 4, ARBITRARY_TAG);
	
	double * threshold= new double[1];
	status=distributed::receive_double_values(threshold, 1, ARBITRARY_TAG);
	distributed::min_threshold=threshold[0];
					
	(configurations[0]==1)?distributed::test=true:distributed::test=false;
	(configurations[1]==1)?distributed::test_send_receive_times=true:distributed::test_send_receive_times=false;
	(configurations[2]==1)?distributed::rpc_time=true:distributed::rpc_time=false;
	(configurations[3]==1)?cache_enabled=true:cache_enabled=false;
			
#ifndef NDEBUG
	if (distributed::test && my_rank==1) distributed::test_parameter_info_slave(proj_pair_sptr->stir::ParsingObject::parameter_info());
#endif			
	//create new Object of objective_function
	PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT > *objective_function = new PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT >();
	shared_ptr<PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT > > objective_function_ptr(objective_function);
			
	//get symmetries pointer
	shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr = 
	  proj_pair_sptr->get_symmetries_used()->clone();
   			
	//set values in objective_function
	objective_function_ptr->set_zero_seg0_end_planes(zero_seg0_end_planes);
	objective_function_ptr->set_projector_pair_sptr(proj_pair_sptr);
			
	//set up ProjData- and binwise_correction objects
	proj_data_ptr = new ProjDataInMemory(proj_data_info_sptr);
	binwise_correction = new ProjDataInMemory(proj_data_info_sptr);
			
	//TODO parallel sensitivity computation
	//Receive normalization_ptr
	//distributed_compute_sensitivity(objective_function_ptr);
			
	distributed_compute_sub_gradient(objective_function_ptr, proj_data_info_sptr, symmetries_sptr);
      }
    else if (obj_fct==100)//objective_function is PoissonLogLikelihoodWithLinearModelForMeanAndListModeData
      {
	error("Parallel List-Mode-Case is not yet implemented. Please use the sequential version!");
      }	
    else warning("Objective function not supported!");		 
  }
   
  template <typename TargetT>
  void DistributedWorker<TargetT>::distributed_compute_sub_gradient(shared_ptr<PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT > > &objective_function_sptr, 
								    const shared_ptr<ProjDataInfo> proj_data_info_sptr, 
								    const shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr)
  {	
    //input- und output-image pointers
    shared_ptr<TargetT> input_image_ptr;
    shared_ptr<TargetT> output_image_ptr;
   		
    int image_buffer_size;
 		
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
    distributed::receive_and_set_image_parameters(input_image_ptr, image_buffer_size, -1, 0);
  			   	   	
    //array to receive count and vs_num values
    int * int_values = new int[2];
    int v_num, s_num;
   		 
    int count, count2;
    HighResWallClockTimer t;
   		
    //loop over the iterations until received END_RECONSTRUCTION_TAG
    while (true)
      {
	//allocate new output_image
	output_image_ptr=input_image_ptr->get_empty_copy();
   		
	//Receive input_image values 
	MPI_Status status = distributed::receive_image_values_and_fill_image_ptr(input_image_ptr, image_buffer_size, 0);
 		   
	//loop to receive viewgrams until received END_ITERATION_TAG
	while (true)
	  {
	    RelatedViewgrams<float>* viewgrams = NULL;
	    RelatedViewgrams<float>* additive_binwise_correction_viewgrams = NULL;
	    count = 0;
	    count2 = 0;
	     		     	
	    //receive vs_num values	
	    status = distributed::receive_int_values(int_values, 2, ARBITRARY_TAG);
	     		
	    v_num = int_values[0];
	    s_num = int_values[1];
	    		
	    /*check whether to
	     *  - use a viewgram already received in previous iteration
	     *  - receive a new viewgram
	     * 	- end the iteration
	     */ 	
	    if (status.MPI_TAG==REUSE_VIEWGRAM_TAG) //use a viewgram already available
	      {
		ViewSegmentNumbers * vs = new ViewSegmentNumbers(v_num, s_num);
	    		
		viewgrams = new RelatedViewgrams<float>(proj_data_ptr->get_related_viewgrams(*vs, symmetries_sptr));
		additive_binwise_correction_viewgrams = new RelatedViewgrams<float>(binwise_correction->get_related_viewgrams(*vs, symmetries_sptr));
		delete vs;
	      }	
	    else if (status.MPI_TAG==NEW_VIEWGRAM_TAG) //receive a message with a new viewgram
	      {
#ifndef NDEBUG
		//run test for related viewgrams
		if (distributed::test && my_rank==1 && distributed::first_iteration==true) distributed::test_related_viewgrams_slave(proj_data_info_sptr, symmetries_sptr);
#endif	    	
		//receive info if additive_binwise_correction_viewgrams are NULL	
		bool add_bin_corr_viewgrams=distributed::receive_bool_value(BINWISE_CORRECTION_TAG, 0);
	    	
		if (add_bin_corr_viewgrams) distributed::receive_and_construct_related_viewgrams(additive_binwise_correction_viewgrams, proj_data_info_sptr, symmetries_sptr, 0);
		distributed::receive_and_construct_related_viewgrams(viewgrams, proj_data_info_sptr, symmetries_sptr, 0); 
	    		
		//save Viewgrams to ProjDataInMemory object
		if(cache_enabled)
		  {
		    if (proj_data_ptr->set_related_viewgrams(*viewgrams)==Succeeded::no)
		      error("Slave %i: Storing viewgrams failed!\n", my_rank);
	    			
		    if (add_bin_corr_viewgrams) 
		      if (binwise_correction->set_related_viewgrams(*additive_binwise_correction_viewgrams)==Succeeded::no)
			error("Slave %i: Storing additive_binwise_correction_viewgrams failed!\n", my_rank);
		  }
	      }
	    else  //the iteration or reconstruction is completed --> send results
	      {	
		//make reduction over computed output_images
		distributed::first_iteration=false;
		distributed::reduce_output_image(output_image_ptr, image_buffer_size, my_rank, 0);
    	  			
		if(distributed::rpc_time)
		  {
		    double * send = new double[1];
		    send[0]=distributed::total_rpc_time;
		    double * receive = new double[1];
		    MPI_Reduce(send, receive, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		    distributed::total_rpc_time = 0.0;
		  }
	      }
	    		
	    //TODO: END_RECONSTRUCTION_TAG is never sent (not necessarily needed)
	    if (status.MPI_TAG==END_ITERATION_TAG) break;
	    else if (status.MPI_TAG==END_RECONSTRUCTION_TAG) return;
	    
	    //measure time used for parallelized part
	    if (distributed::rpc_time) {t.reset(); t.start();}
	    		
	    //call the actual Reconstruction
	    RPC_process_related_viewgrams_gradient(
						   objective_function_sptr->get_projector_pair_sptr()->get_forward_projector_sptr(),
						   objective_function_sptr->get_projector_pair_sptr()->get_back_projector_sptr(),
						   output_image_ptr.get(), input_image_ptr.get(), viewgrams, 
						   count, count2,
						   log_likelihood_ptr,
						   additive_binwise_correction_viewgrams);
				
	    if (distributed::rpc_time)
	      {
		t.stop();
		distributed::total_rpc_time=distributed::total_rpc_time + t.value();
		distributed::total_rpc_time_2=distributed::total_rpc_time_2 + t.value();
	      } 
				
	    int_values[0]=count;
	    int_values[1]=count2;
															
	    //send count,count2 and ask for new work
	    distributed::send_int_values(int_values, 2, AVAILABLE_NOTIFICATION_TAG, 0);
    	  		
	    if (viewgrams!=NULL) delete viewgrams;
	    if (additive_binwise_correction_viewgrams!=NULL) delete additive_binwise_correction_viewgrams;		
	  }
	if (distributed::rpc_time)
	  cerr << "Slave "<<my_rank<<" used "<<distributed::total_rpc_time_2<<" seconds for PRC-processing."<<endl;
      }     	
    delete int_values;
  }
  
  template <typename TargetT>
  void DistributedWorker<TargetT>::distributed_compute_sensitivity(shared_ptr<PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT > > &objective_function_ptr)
  {
    //TODO implement parallel version of sensitivity computation
  }
   
  template class DistributedWorker<DiscretisedDensity<3,float> >;  
   
} // namespace stir
