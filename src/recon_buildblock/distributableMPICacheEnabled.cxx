//
// $Id$
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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

  \brief Implementation of stir::distributable_computation_cache_enabled()

  \author Kris Thielemans  
  \author Alexey Zverovich 
  \author Matthew Jacobson
  \author PARAPET project
  \author Tobias Beisel

  $Date$
  $Revision$
*/
/* Modification history:
   KT 30/05/2002
   get rid of dependence on specific symmetries (i.e. views up to 45 degrees)
   
   TB 30/06/2007
   added MPI support
   */
#include "stir/shared_ptr.h"
#include "stir/recon_buildblock/distributableMPICacheEnabled.h"
#include "stir/RelatedViewgrams.h"
#include "stir/ProjData.h"
#include "stir/DiscretisedDensity.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/CPUTimer.h"
#include "stir/recon_buildblock/ForwardProjectorByBin.h"
#include "stir/recon_buildblock/BackProjectorByBin.h"

#ifdef STIR_MPI
#include "stir/recon_buildblock/distributed_functions.h"
#include "stir/recon_buildblock/distributed_test_functions.h"
#include "stir/recon_buildblock/DistributedCachingInformation.h"
#endif

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif


START_NAMESPACE_STIR

void distributable_computation_cache_enabled(
					     const shared_ptr<ForwardProjectorByBin>& forward_projector_ptr,
					     const shared_ptr<BackProjectorByBin>& back_projector_ptr,
					     const shared_ptr<DataSymmetriesForViewSegmentNumbers>& symmetries_ptr,
					     DiscretisedDensity<3,float>* output_image_ptr,
					     const DiscretisedDensity<3,float>* input_image_ptr,
					     const shared_ptr<ProjData>& proj_dat_ptr, 
					     const bool read_from_proj_dat,
					     int subset_num, int num_subsets,
					     int min_segment_num, int max_segment_num,
					     bool zero_seg0_end_planes,
					     double* log_likelihood_ptr,
					     const shared_ptr<ProjData>& binwise_correction,
					     RPC_process_related_viewgrams_type * RPC_process_related_viewgrams, 
					     DistributedCachingInformation* caching_info_ptr
					     )
{ 
  //test distributed functions (see DistributedTestFunctions.h for details)
#ifndef NDEBUG
  if (distributed::test && distributed::first_iteration) 
    {
      distributed::test_image_estimate_master(input_image_ptr, 1);
      distributed::test_parameter_info_master(proj_dat_ptr->get_proj_data_info_ptr()->parameter_info(), 1, "proj_data_info");
      distributed::test_bool_value_master(true, 1);
      distributed::test_int_value_master(444, 1);
      distributed::test_int_values_master(1);
		
      Viewgram<float>* viewgram = new Viewgram<float>(const_cast<ProjDataInfo*>(proj_dat_ptr->get_proj_data_info_ptr()), 44, 0);
      for ( int tang_pos = viewgram->get_min_tangential_pos_num(); tang_pos  <= viewgram->get_max_tangential_pos_num() ;++tang_pos)  
	for ( int ax_pos = viewgram->get_min_axial_pos_num(); ax_pos <= viewgram->get_max_axial_pos_num() ;++ax_pos)
	  (*viewgram)[ax_pos][tang_pos]= rand();
				
      distributed::test_viewgram_master(*viewgram, const_cast<ProjDataInfo*>(proj_dat_ptr->get_proj_data_info_ptr()));
		
      viewgram=NULL;		
      delete viewgram;
    }
#endif
	
  //send the current image estimate
  if (distributed::first_iteration == true) distributed::send_image_parameters(input_image_ptr, -1, -1);
  distributed::send_image_estimate(input_image_ptr, -1);
  
  assert(min_segment_num <= max_segment_num);
  assert(subset_num >=0);
  assert(subset_num < num_subsets);
  
  assert(proj_dat_ptr.use_count() != 0);
  	 
  	
  if (output_image_ptr != NULL)
    output_image_ptr->fill(0);
  
  if (log_likelihood_ptr != NULL)
    {
      (*log_likelihood_ptr) = 0.0;
    };
  	
  // needed for several send/receive operations
  int * int_values = new int[2];
	
  int processed_count=0;		//counts the number of processed vs_nums 
  int sent_count=0;			//counts the work packages sent
  int working_slaves_count=0; //counts the number of slaves which are currently working
  int next_receiver=1;		//always stores the next slave to be provided with work
	
  int current_segment=-1;		//value which stores the segment currently processed
	
  int count=0, count2=0;
  bool new_viewgrams=true;	//whether the vs_num currently processed is a new or an already cached one for the slave 
	
  //TODO: reconsider these values!!! Tested to be fine
  //calculate number of views and number of segments
  int num_segs = max_segment_num - min_segment_num +1;
  int num_views = ((int)((proj_dat_ptr->get_max_view_num() - proj_dat_ptr->get_min_view_num() - subset_num)/num_subsets))+1;
  int num_vs = num_segs * num_views; 
	
  CPUTimer iteration_timer;
  iteration_timer.start();
   
  //initialize the caching values for the new iteration
  if (distributed::first_iteration==false) caching_info_ptr->initialize_new_iteration();
  distributed::iteration_counter++;
    
  int segment_num=0;
  int view=0;
    
  //while not all vs_nums are processed repeat
  while (processed_count < num_vs)
    {     	
      ViewSegmentNumbers view_segment_num;
		
      //if this is the case, all data has been distributed, so the caching algorithm can be used 
      if (distributed::iteration_counter>num_subsets) 
	{
	  //check whether the slave will receive a new or an already cached viewgram
	  if (caching_info_ptr->get_remaining_vs_nums(next_receiver, subset_num)==0) new_viewgrams=true;
	  else new_viewgrams=false;
  		 
	  //use stored caching information to get the next unprocessed vs_num for the slave
	  view_segment_num = caching_info_ptr->get_unprocessed_vs_num(next_receiver, subset_num);
  			
	  view = view_segment_num.view_num();
	  segment_num = view_segment_num.segment_num();
	}
      //the non cached way of distributing a vs_num
      else 
	{
	  //TODO: reconsider these values!!! Tested to be fine
	  //calculate current segment (kind of ceiling function)
	  double s = processed_count/num_views;
	  int x = (int)s; // complete segments 
	  double y = s - x; // part of current segment (0 if not yet next)
  			
	  if (y>0.0) segment_num = x+1;
	  else segment_num = min_segment_num + x;
  			
	  //calculate view
	  if (x>current_segment) 
	    {
	      view = proj_dat_ptr->get_min_view_num() + subset_num;
	      current_segment=x;
	    }
	  else view =  view + num_subsets;
			
	  //TODO this is kind of wasted memory  
	  ViewSegmentNumbers temp_view_segment_num(view, segment_num);
	  view_segment_num = temp_view_segment_num;
	  new_viewgrams=true;
      		
	  //add information about the distributed values to the caching object
	  if (symmetries_ptr->is_basic(view_segment_num))
	    {
	      caching_info_ptr->add_proc_to_vs_num(view_segment_num, next_receiver);
	      caching_info_ptr->add_vs_num_to_proc(next_receiver, view_segment_num, subset_num);
	    }
	}	
  		
      //all work done
      if (view==-1 && segment_num==-1) break;
  		
      processed_count++;    
  		
      if (!symmetries_ptr->is_basic(view_segment_num))
	continue;
  		
      //the slave has not yet processed this vs_num, so the viewgrams have to be sent			
      if (new_viewgrams==true)
	{	
	  int_values[0]=view;
	  int_values[1]=segment_num;

	  //send the vs_num with new_viewgram-tag
	  distributed::send_int_values(int_values, 2, NEW_VIEWGRAM_TAG, next_receiver);
			
	  const ViewSegmentNumbers view_segment_num(view, segment_num);
			
	  RelatedViewgrams<float>* additive_binwise_correction_viewgrams = NULL;
			
	  if (binwise_correction.use_count() != 0) 
	    {
#ifndef _MSC_VER
	      additive_binwise_correction_viewgrams =
		new RelatedViewgrams<float>
		(binwise_correction->get_related_viewgrams(view_segment_num, symmetries_ptr));
#else
	      RelatedViewgrams<float> tmp(binwise_correction->
					  get_related_viewgrams(view_segment_num, symmetries_ptr));
	      additive_binwise_correction_viewgrams = new RelatedViewgrams<float>(tmp);
#endif      
	    }
			
	  RelatedViewgrams<float>* y = NULL;
					      
	  if (read_from_proj_dat)
	    {
#ifndef _MSC_VER
	      y = new RelatedViewgrams<float>
		(proj_dat_ptr->get_related_viewgrams(view_segment_num, symmetries_ptr));
#else
	      // workaround VC++ 6.0 bug
	      RelatedViewgrams<float> tmp(proj_dat_ptr->
					  get_related_viewgrams(view_segment_num, symmetries_ptr));
	      y = new RelatedViewgrams<float>(tmp);
#endif        
	    }
	  else
	    {
	      y = new RelatedViewgrams<float>
		(proj_dat_ptr->get_empty_related_viewgrams(view_segment_num, symmetries_ptr));
	    }
						      
#ifndef NDEBUG
	  // test if symmetries didn't take us out of the segment range
	  for (RelatedViewgrams<float>::iterator r_viewgrams_iter = y->begin();
	       r_viewgrams_iter != y->end();
	       ++r_viewgrams_iter)
	    {
	      assert(r_viewgrams_iter->get_segment_num() >= min_segment_num);
	      assert(r_viewgrams_iter->get_segment_num() <= max_segment_num);
	    }
#endif     
			
	  if (segment_num==0 && zero_seg0_end_planes)
	    {
	      if (y != NULL)
		{
		  const int min_ax_pos_num = y->get_min_axial_pos_num();
		  const int max_ax_pos_num = y->get_max_axial_pos_num();
		  for (RelatedViewgrams<float>::iterator r_viewgrams_iter = y->begin();
		       r_viewgrams_iter != y->end();
		       ++r_viewgrams_iter)
		    {
		      (*r_viewgrams_iter)[min_ax_pos_num].fill(0);
		      (*r_viewgrams_iter)[max_ax_pos_num].fill(0);
		    }
		};
					       
	      if (additive_binwise_correction_viewgrams != NULL)
		{
		  const int min_ax_pos_num = additive_binwise_correction_viewgrams->get_min_axial_pos_num();
		  const int max_ax_pos_num = additive_binwise_correction_viewgrams->get_max_axial_pos_num();
		  for (RelatedViewgrams<float>::iterator r_viewgrams_iter = additive_binwise_correction_viewgrams->begin();
		       r_viewgrams_iter != additive_binwise_correction_viewgrams->end();
		       ++r_viewgrams_iter)
		    {
		      (*r_viewgrams_iter)[min_ax_pos_num].fill(0);
		      (*r_viewgrams_iter)[max_ax_pos_num].fill(0);
		    }
		}
	    }

#ifndef NDEBUG
	  //test sending related viegrams
	  if (distributed::test && distributed::first_iteration==true && next_receiver==1) 
	    distributed::test_related_viewgrams_master(const_cast<stir::ProjDataInfo*>(y->get_proj_data_info_ptr()), symmetries_ptr, y, next_receiver);
#endif		
	  //TODO: this could also be done by using MPI_Probe at the slave to find out what to recieve next
	  if (additive_binwise_correction_viewgrams == NULL)
	    {	//tell slaves that recieving additive_binwise_correction_viewgrams is not needed
	      distributed::send_bool_value(false, BINWISE_CORRECTION_TAG, next_receiver);
	    }
	  else
	    {
	      //tell slaves to receive additive_binwise_correction_viewgrams
	      distributed::send_bool_value(true, BINWISE_CORRECTION_TAG, next_receiver);
	      distributed::send_related_viewgrams(additive_binwise_correction_viewgrams, next_receiver);
	    }
	   		
	  //this gives the slave the actual work to do      	 
	  distributed::send_related_viewgrams(y, next_receiver);
	  working_slaves_count++;
	  sent_count++;
		
	  MPI_Status status;
		
	  //give every slave some work before waiting for requests 
	  if (sent_count <distributed::num_processors-1) next_receiver++; 
	  else 
	    {	
	      status=distributed::receive_int_values(int_values, 2, AVAILABLE_NOTIFICATION_TAG);
	      next_receiver=status.MPI_SOURCE;
	      working_slaves_count--;
				
	      //reduce count values
	      count+=int_values[0];
	      count2+=int_values[1];
	    }
			
	  if (additive_binwise_correction_viewgrams != NULL)
	    {
	      delete additive_binwise_correction_viewgrams;
	      additive_binwise_correction_viewgrams = NULL;
	    };
	  delete y;
	}
      else //use cached viewgram 
	{
	  MPI_Status status;
			
	  int_values[0] = view;
	  int_values[1] = segment_num;
			
	  //send vs_num with reuse-tag
	  distributed::send_int_values(int_values, 2, REUSE_VIEWGRAM_TAG, next_receiver);
			
	  working_slaves_count++;
	  sent_count++;
			
	  if (sent_count <distributed::num_processors-1) next_receiver++; 
	  else 
	    {	
	      status=distributed::receive_int_values(int_values, 2, AVAILABLE_NOTIFICATION_TAG);
	      next_receiver=status.MPI_SOURCE;
	      working_slaves_count--;
				
	      //reduce count values
	      count+=int_values[0];
	      count2+=int_values[1];
	    }
	}
		
    }

  distributed::first_iteration=false;
	
  //receive remaining available notifications
  MPI_Status status;
  while(working_slaves_count>0)
    {
      status=distributed::receive_int_values(int_values, 2, AVAILABLE_NOTIFICATION_TAG);
      working_slaves_count--;
		
      //reduce count values
      count+=int_values[0];
      count2+=int_values[1];
    }
	
  // in the cache-enabled distributed case, this message is only printed once per iteration
  // TODO this message relies on knowledge of count, count2 which might be inappropriate for 
  // the call-back function
  cerr<<"\tNumber of (cancelled) singularities: "<<count
      <<"\n\tNumber of (cancelled) negative numerators: "<<count2
      << "\n\tIteration: "<< iteration_timer.value() << "secs" <<endl;
    
  int_values[0]=3;
  int_values[1]=4;
   
  //send end_iteration notification
  distributed::send_int_values(int_values, 2, END_ITERATION_TAG, -1);
	
  //reduce output_image
  distributed::reduce_received_output_image(output_image_ptr, 0);
	
  //reduce timed rpc-value
  if(distributed::rpc_time)
    {
      printf("Master: Reducing timer value\n");
      double * send = new double[1];
      double * receive = new double[1];
      MPI_Reduce(send, receive, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      distributed::total_rpc_time+=(receive[0]/(distributed::num_processors-1));
	
      printf("Average time used by slaves for RPC processing: %f secs\n", distributed::total_rpc_time);
      distributed::total_rpc_time_slaves+=receive[0];
    }
	
  delete[] int_values;
}

END_NAMESPACE_STIR
