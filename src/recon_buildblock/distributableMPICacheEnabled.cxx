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
  \todo merge with distributable.cxx

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
   KT 2011
   some corrections to TB's work
   */
#include "stir/shared_ptr.h"
#include "stir/is_null_ptr.h"
#include "stir/recon_buildblock/distributableMPICacheEnabled.h"
#include "stir/RelatedViewgrams.h"
#include "stir/ProjData.h"
#include "stir/DiscretisedDensity.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/CPUTimer.h"
#include "stir/recon_buildblock/ForwardProjectorByBin.h"
#include "stir/recon_buildblock/BackProjectorByBin.h"
#include "stir/recon_buildblock/BinNormalisation.h"
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
// TODO all these functions are the same as in distributable.cxx
// we really just need to move a few things from this file to distributable.cxx, and then get rid of this one

template <class ViewgramsPtr>
static void
zero_end_sinograms(ViewgramsPtr viewgrams_ptr)
{
  if (!is_null_ptr(viewgrams_ptr))
    {
      const int min_ax_pos_num = viewgrams_ptr->get_min_axial_pos_num();
      const int max_ax_pos_num = viewgrams_ptr->get_max_axial_pos_num();
      for (RelatedViewgrams<float>::iterator r_viewgrams_iter = viewgrams_ptr->begin();
           r_viewgrams_iter != viewgrams_ptr->end();
           ++r_viewgrams_iter)
        {
          (*r_viewgrams_iter)[min_ax_pos_num].fill(0);
          (*r_viewgrams_iter)[max_ax_pos_num].fill(0);
        }
    }
}

static std::vector<ViewSegmentNumbers> 
find_vs_nums_in_subset(const ProjDataInfo& proj_data_info,
                       const DataSymmetriesForViewSegmentNumbers& symmetries, 
                       const int min_segment_num, const int max_segment_num,
                       const int subset_num, const int num_subsets)
{
  std::vector<ViewSegmentNumbers> vs_nums_to_process;
  for (int segment_num = min_segment_num; segment_num <= max_segment_num; segment_num++)
  {
    for (int view = proj_data_info.get_min_view_num() + subset_num; 
        view <= proj_data_info.get_max_view_num(); 
        view += num_subsets)
    {
      const ViewSegmentNumbers view_segment_num(view, segment_num);
        
      if (!symmetries.is_basic(view_segment_num))
        continue;

      vs_nums_to_process.push_back(view_segment_num);

#ifndef NDEBUG
      // test if symmetries didn't take us out of the segment range
      std::vector<ViewSegmentNumbers> rel_vs;
      symmetries.get_related_view_segment_numbers(rel_vs, view_segment_num);
      for (std::vector<ViewSegmentNumbers>::const_iterator iter = rel_vs.begin(); iter!= rel_vs.end(); ++iter)
        {
          assert(iter->segment_num() >= min_segment_num);
          assert(iter->segment_num() <= max_segment_num);
        }
#endif
    }
  }
  return vs_nums_to_process;
}

static
void get_viewgrams(shared_ptr<RelatedViewgrams<float> >& y,
                   shared_ptr<RelatedViewgrams<float> >& additive_binwise_correction_viewgrams,
                   shared_ptr<RelatedViewgrams<float> >& mult_viewgrams_sptr,
                   const shared_ptr<ProjData>& proj_dat_ptr, 
                   const bool read_from_proj_dat,
                   const bool zero_seg0_end_planes,
                   const shared_ptr<ProjData>& binwise_correction,
                   const shared_ptr<BinNormalisation>& normalisation_sptr,
                   const double start_time_of_frame,
                   const double end_time_of_frame,
                   const shared_ptr<DataSymmetriesForViewSegmentNumbers>& symmetries_ptr,
                   const ViewSegmentNumbers& view_segment_num
                   )
{
  if (!is_null_ptr(binwise_correction)) 
    {
#if !defined(_MSC_VER) || _MSC_VER>1300
      additive_binwise_correction_viewgrams =
        new RelatedViewgrams<float>
        (binwise_correction->get_related_viewgrams(view_segment_num, symmetries_ptr));
#else
      RelatedViewgrams<float> tmp(binwise_correction->
                                  get_related_viewgrams(view_segment_num, symmetries_ptr));
      additive_binwise_correction_viewgrams = new RelatedViewgrams<float>(tmp);
#endif      
    }
                        
  if (read_from_proj_dat)
    {
#if !defined(_MSC_VER) || _MSC_VER>1300
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

  // multiplicative correction
  if (!is_null_ptr(normalisation_sptr) && !normalisation_sptr->is_trivial())
    {
      mult_viewgrams_sptr =
        new RelatedViewgrams<float>(proj_dat_ptr->get_empty_related_viewgrams(view_segment_num, symmetries_ptr));
      mult_viewgrams_sptr->fill(1.F);
      normalisation_sptr->undo(*mult_viewgrams_sptr,start_time_of_frame,end_time_of_frame);
    }
                        
  if (view_segment_num.segment_num()==0 && zero_seg0_end_planes)
    {
      zero_end_sinograms(y);
      zero_end_sinograms(additive_binwise_correction_viewgrams);
      zero_end_sinograms(mult_viewgrams_sptr);
    }
}

static void send_viewgrams(const shared_ptr<RelatedViewgrams<float> >& y,
                    const shared_ptr<RelatedViewgrams<float> >& additive_binwise_correction_viewgrams,
                    const shared_ptr<RelatedViewgrams<float> >& mult_viewgrams_sptr,
                    const int next_receiver)
{
  distributed::send_view_segment_numbers( y->get_basic_view_segment_num(), NEW_VIEWGRAM_TAG, next_receiver);

#ifndef NDEBUG
  //test sending related viegrams
  if (distributed::test && distributed::first_iteration==true && next_receiver==1) 
    distributed::test_related_viewgrams_master(y->get_proj_data_info_ptr()->clone(), 
                                               y->get_symmetries_ptr()->clone(), y.get(), next_receiver);
#endif

  //TODO: this could also be done by using MPI_Probe at the slave to find out what to recieve next
  if (is_null_ptr(additive_binwise_correction_viewgrams))
    {   //tell slaves that recieving additive_binwise_correction_viewgrams is not needed
      distributed::send_bool_value(false, BINWISE_CORRECTION_TAG, next_receiver);
    }
  else
    {
      //tell slaves to receive additive_binwise_correction_viewgrams
      distributed::send_bool_value(true, BINWISE_CORRECTION_TAG, next_receiver);
      distributed::send_related_viewgrams(additive_binwise_correction_viewgrams.get(), next_receiver);
    }
  if (is_null_ptr(mult_viewgrams_sptr))
    {
      //tell slaves that recieving mult_viewgrams is not needed
      distributed::send_bool_value(false, BINWISE_MULT_TAG, next_receiver);
    }
  else
    {
      //tell slaves to receive mult_viewgrams
      distributed::send_bool_value(true, BINWISE_MULT_TAG, next_receiver);
      distributed::send_related_viewgrams(mult_viewgrams_sptr.get(), next_receiver);
    }
  // send y
  distributed::send_related_viewgrams(y.get(), next_receiver);
}

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
                                             const shared_ptr<BinNormalisation> normalise_sptr,
                                             const double start_time_of_frame,
                                             const double end_time_of_frame,
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
                
      Viewgram<float> viewgram(proj_dat_ptr->get_proj_data_info_ptr()->clone(), 44, 0);
      for ( int tang_pos = viewgram.get_min_tangential_pos_num(); tang_pos  <= viewgram.get_max_tangential_pos_num() ;++tang_pos)  
        for ( int ax_pos = viewgram.get_min_axial_pos_num(); ax_pos <= viewgram.get_max_axial_pos_num() ;++ax_pos)
          viewgram[ax_pos][tang_pos]= rand();
                                
      distributed::test_viewgram_master(viewgram, proj_dat_ptr->get_proj_data_info_ptr()->clone());
    }
#endif
        
  //send the current image estimate
  distributed::send_image_estimate(input_image_ptr, -1);
  
  assert(min_segment_num <= max_segment_num);
  assert(subset_num >=0);
  assert(subset_num < num_subsets);
  
  assert(!is_null_ptr(proj_dat_ptr));  
  if (output_image_ptr != NULL)
    output_image_ptr->fill(0);
  
  if (log_likelihood_ptr != NULL)
    {
      (*log_likelihood_ptr) = 0.0;
    };
        
  // needed for several send/receive operations
  int int_values[2];
        
  int working_slaves_count=0; //counts the number of slaves which are currently working
  int next_receiver=1;          //always stores the next slave to be provided with work
        
  int count=0, count2=0;
        
  const std::vector<ViewSegmentNumbers> vs_nums_to_process = 
    find_vs_nums_in_subset(*proj_dat_ptr->get_proj_data_info_ptr(), *symmetries_ptr,
                           min_segment_num, max_segment_num,
                           subset_num, num_subsets);
  
  const std::size_t num_vs = vs_nums_to_process.size(); 
        
  CPUTimer iteration_timer;
  iteration_timer.start();
   
  //initialize the caching values for the new iteration
  caching_info_ptr->initialise_new_subiteration(vs_nums_to_process);
    
  //while not all vs_nums are processed repeat
  for (std::size_t processed_count = 1; processed_count <= num_vs; ++processed_count)
    {           
      ViewSegmentNumbers view_segment_num;
                
      //check whether the slave will receive a new or an already cached viewgram
      const bool new_viewgrams = 
         caching_info_ptr->get_unprocessed_vs_num(view_segment_num, next_receiver);
      // view_segment_num = vs_nums_to_process[processed_count-1];
                                
      //the slave has not yet processed this vs_num, so the viewgrams have to be sent                   
      if (new_viewgrams==true)
        {       
          cerr << "Sending segment " << view_segment_num.segment_num() 
               << ", view " << view_segment_num.view_num() << " to slave " << next_receiver << std::endl;

          shared_ptr<RelatedViewgrams<float> > y = 0;                                         
          shared_ptr<RelatedViewgrams<float> > additive_binwise_correction_viewgrams = 0;
          shared_ptr<RelatedViewgrams<float> > mult_viewgrams_sptr = 0;

          get_viewgrams(y, additive_binwise_correction_viewgrams, mult_viewgrams_sptr,
                        proj_dat_ptr, read_from_proj_dat,
                        zero_seg0_end_planes,
                        binwise_correction,
                        normalise_sptr, start_time_of_frame, end_time_of_frame,
                        symmetries_ptr, view_segment_num);

          //send viewgrams, the slave will immediatelly start calculation
          send_viewgrams(y, additive_binwise_correction_viewgrams, mult_viewgrams_sptr,
                         next_receiver);
        } // if(new_viewgram)
      else
        {
          cerr << "Re-using  segment " << view_segment_num.segment_num() 
               << ", view " << view_segment_num.view_num() << " at slave " << next_receiver << std::endl;
          //send vs_num with reuse-tag, the slave will immediatelly start calculation
          distributed::send_view_segment_numbers( view_segment_num, REUSE_VIEWGRAM_TAG, next_receiver);
        }

      working_slaves_count++;
                      
      if (int(processed_count) <distributed::num_processors-1) // note: -1 as master doesn't get any viewgrams
        {
          //give every slave some work before waiting for requests 
          next_receiver++; 
        }
      else 
        {       
          const MPI_Status status=distributed::receive_int_values(int_values, 2, AVAILABLE_NOTIFICATION_TAG);
          next_receiver=status.MPI_SOURCE;
          working_slaves_count--;
                                
          //reduce count values
          count+=int_values[0];
          count2+=int_values[1];
        }               
    } // end loop over vs_nums 

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
#ifndef STIR_NO_RUNNING_LOG
  cerr<<"\tNumber of (cancelled) singularities: "<<count
      <<"\n\tNumber of (cancelled) negative numerators: "<<count2
      << "\n\tIteration: "<< iteration_timer.value() << "secs" <<endl;
#endif
    
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
      double send = 0;
      double receive;
      MPI_Reduce(&send, &receive, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      distributed::total_rpc_time+=(receive/(distributed::num_processors-1));
        
      printf("Average time used by slaves for RPC processing: %f secs\n", distributed::total_rpc_time);
      distributed::total_rpc_time_slaves+=receive;
    }
        
}

END_NAMESPACE_STIR
