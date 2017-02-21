//
//
/*!

  \file
  \ingroup listmode
  \brief Class stir::LmToProjDataBootstrap for rebinning listmode files with the bootstrap method
    
  \author Kris Thielemans
      
*/
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
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

#include "stir/listmode/LmToProjDataBootstrap.h"
#include "stir/listmode/CListRecord.h"
#include "stir/Succeeded.h"
#include "stir/info.h"
#include <iostream>
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif

#include <boost/random/uniform_int.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>



START_NAMESPACE_STIR

template <typename LmToProjDataT>
void 
LmToProjDataBootstrap<LmToProjDataT>::set_defaults()
{
  LmToProjData::set_defaults();
  seed = 42;
}

template <typename LmToProjDataT>
void 
LmToProjDataBootstrap<LmToProjDataT>::initialise_keymap()
{
  LmToProjData::initialise_keymap();
  this->parser.add_start_key("LmToProjDataBootstrap Parameters");
  this->parser.add_key("seed", reinterpret_cast<int *>(&seed)); // TODO get rid of cast
}

template <typename LmToProjDataT>
LmToProjDataBootstrap<LmToProjDataT>::
LmToProjDataBootstrap(const char * const par_filename)
{
  set_defaults();
  if (par_filename!=0)
    this->parse(par_filename) ;
  else
    this->ask_parameters();
}

template <typename LmToProjDataT>
LmToProjDataBootstrap<LmToProjDataT>::
LmToProjDataBootstrap(const char * const par_filename, const unsigned int seed_v)
{
  set_defaults();
  seed = seed_v;
  if (par_filename!=0)
    {
      this->parse(par_filename);
      // make sure that seed_v parameter overrides whatever was in the par file
      if (seed != seed_v)
	{
	  warning("LmToProjDataBootstrap: parameter file %s contains seed (%u) which is\n"
		  "different from the seed value (%u) passed to me.\n"
		  "I will use the latter.\n",
		  par_filename, seed, seed_v);
	  seed = seed_v;
	}
    }
  else
    this->ask_parameters();
}

template <typename LmToProjDataT>
bool
LmToProjDataBootstrap<LmToProjDataT>::
post_processing()
{
  if (LmToProjData::post_processing())
    return true;

  if (seed == 0)
    return true;

  return false;
}



template <typename LmToProjDataT> 
void 
LmToProjDataBootstrap<LmToProjDataT>::
start_new_time_frame(const unsigned int new_frame_num)
{

  base_type::start_new_time_frame(new_frame_num);

  const double start_time = this->frame_defs.get_start_time(new_frame_num);
  const double end_time = this->frame_defs.get_end_time(new_frame_num);
  // When do_time_frame=true, the number of events is irrelevant, so we 
  // just set more_events to 1, and never change it
  long more_events = 
    this->do_time_frame? 1 : this->num_events_to_store;

  unsigned int total_num_events_in_this_frame = 0;

  // loop over all events in the listmode file
  shared_ptr <CListRecord> record_sptr = this->lm_data_ptr->get_empty_record_sptr();
  CListRecord& record = *record_sptr;

  info("Going through listmode file to find number of events in this frame");
  double current_time = start_time;
  while (more_events)
    {
      if (this->lm_data_ptr->get_next_record(record) == Succeeded::no) 
	{
	  // no more events in file for some reason
	  break; //get out of while loop
	}
      if (record.is_time())
	{
	  const double new_time = record.time().get_time_in_secs();
	  if (this->do_time_frame && new_time >= end_time)
	    break; // get out of while loop
	  current_time = new_time;
	}
      else if (record.is_event() && start_time <= current_time)
	{
	  ++total_num_events_in_this_frame;

	  if (!this->do_time_frame)
	    {
	      // painful business to decrement more_events

	      // TODO optimisation possible: 
	      // if we reject an event below, we could force its replication count to 0
	      // That way, we will not call get_bin_from_event for it anymore.

	      Bin bin;
	      // set value in case the event decoder doesn't touch it
	      // otherwise it would be 0 and all events will be ignored
	      bin.set_bin_value(1);
          base_type::get_bin_from_event(bin, record.event());
	      // check if it's inside the range we want to store
	      if (bin.get_bin_value()>0
		  && bin.tangential_pos_num()>= this->template_proj_data_info_ptr->get_min_tangential_pos_num()
		  && bin.tangential_pos_num()<= this->template_proj_data_info_ptr->get_max_tangential_pos_num()
		  && bin.axial_pos_num()>=this->template_proj_data_info_ptr->get_min_axial_pos_num(bin.segment_num())
		  && bin.axial_pos_num()<=this->template_proj_data_info_ptr->get_max_axial_pos_num(bin.segment_num())
		  && bin.segment_num()>=this->template_proj_data_info_ptr->get_min_segment_num()
		  && bin.segment_num()<=this->template_proj_data_info_ptr->get_max_segment_num()
		  ) 
		{
		  assert(bin.view_num()>=this->template_proj_data_info_ptr->get_min_view_num());
		  assert(bin.view_num()<=this->template_proj_data_info_ptr->get_max_view_num());
            
		  // see if we increment or decrement the value in the sinogram
		  const int event_increment =
		    record.event().is_prompt() 
		    ? ( this->store_prompts ? 1 : 0 ) // it's a prompt
		    :  this->delayed_increment;//it is a delayed-coincidence event
            
		  if (event_increment==0)
		    continue;
            

		  more_events-= event_increment;
		}
	    } // !do_time_frame
	} // if (record.is_event())
    } // while (more_events)

      // now initialise num_times_to_replicate

  typedef boost::mt19937 base_generator_type;
  base_generator_type generator;    
  generator.seed(static_cast<boost::uint32_t>(seed));

  boost::uniform_int<unsigned> 
    uniform_int_distribution(0U, total_num_events_in_this_frame-1);
  boost::variate_generator<base_generator_type, boost::uniform_int<unsigned> > 
    random_int(generator,
	       uniform_int_distribution);
    
  num_times_to_replicate.resize(total_num_events_in_this_frame);
    
  std::fill( num_times_to_replicate.begin(),  num_times_to_replicate.end(), 
	     static_cast<unsigned char>(0));
  for (unsigned int i=total_num_events_in_this_frame; i!=0; --i)
    {
      const unsigned int event_num = random_int();
      num_times_to_replicate[event_num] += 1;
      // warning this did not check for overflow
    }

  assert(std::accumulate(num_times_to_replicate.begin(),  
			 num_times_to_replicate.end(),
			 0U) == 
	 total_num_events_in_this_frame);
	 
  num_times_to_replicate_iter = num_times_to_replicate.begin();
    
  info(boost::format("Filled in replication vector for %1% events.") % total_num_events_in_this_frame);
}

template <typename LmToProjDataT> 
void 
LmToProjDataBootstrap<LmToProjDataT>::
get_bin_from_event(Bin& bin, const CListEvent& event) const
{
  assert(num_times_to_replicate_iter != num_times_to_replicate.end());
  if (*num_times_to_replicate_iter > 0)
    {
      base_type::get_bin_from_event(bin, event);
      bin.set_bin_value(bin.get_bin_value() * *num_times_to_replicate_iter);
    }
  else
    bin.set_bin_value(-1);
  ++num_times_to_replicate_iter;
}


// instantiation
template class LmToProjDataBootstrap<LmToProjData>;



END_NAMESPACE_STIR
