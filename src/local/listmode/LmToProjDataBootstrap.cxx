//
// $Id$
//
/*!

  \file
  \ingroup listmode
  \brief Class for rebinning listmode files with the bootstrap method
    
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "local/stir/listmode/LmToProjDataBootstrap.h"
#include "local/stir/listmode/CListRecord.h"
#include "stir/Succeeded.h"
#include <iostream>
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif

#ifdef _MSC_VER
// Current version of boost::random breaks on VC6 and 7 because of 
// compile time asserts. I'm disabling them for now by defining the following.
//TODO remove when upgrading to next version
#define BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS 
#endif
#include <boost/random/uniform_int.hpp>
#include <boost/random/mersenne_twister.hpp>



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
  parser.add_start_key("LmToProjDataBootstrap Parameters");
  parser.add_key("seed", reinterpret_cast<int *>(&seed)); // TODO get rid of cast
}

template <typename LmToProjDataT>
LmToProjDataBootstrap<LmToProjDataT>::
LmToProjDataBootstrap(const char * const par_filename)
{
  set_defaults();
  if (par_filename!=0)
    parse(par_filename) ;
  else
    ask_parameters();
}

template <typename LmToProjDataT>
LmToProjDataBootstrap<LmToProjDataT>::
LmToProjDataBootstrap(const char * const par_filename, const unsigned int seed_v)
{
  set_defaults();
  seed = seed_v;
  if (par_filename!=0)
    {
      parse(par_filename);
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
    ask_parameters();
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

  const double start_time = frame_defs.get_start_time(new_frame_num);
  const double end_time = frame_defs.get_end_time(new_frame_num);
  // When do_time_frame=true, the number of events is irrelevant, so we 
  // just set more_events to 1, and never change it
  long more_events = 
    do_time_frame? 1 : num_events_to_store;

  unsigned int total_num_events_in_this_frame = 0;

  // loop over all events in the listmode file
  shared_ptr <CListRecord> record_sptr = lm_data_ptr->get_empty_record_sptr();
  CListRecord& record = *record_sptr;

  cerr << "\nGoing through listmode file to find number of events in this frame" << endl;
  double current_time = start_time;
  while (more_events)
    {
      if (lm_data_ptr->get_next_record(record) == Succeeded::no) 
	{
	  // no more events in file for some reason
	  break; //get out of while loop
	}
      if (record.is_time())
	{
	  const double new_time = record.time().get_time_in_secs();
	  if (do_time_frame && new_time >= end_time)
	    break; // get out of while loop
	  current_time = new_time;
	}
      else if (record.is_event() && start_time <= current_time)
	{
	  ++total_num_events_in_this_frame;

	  if (!do_time_frame)
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
		  && bin.tangential_pos_num()>= template_proj_data_info_ptr->get_min_tangential_pos_num()
		  && bin.tangential_pos_num()<= template_proj_data_info_ptr->get_max_tangential_pos_num()
		  && bin.axial_pos_num()>=template_proj_data_info_ptr->get_min_axial_pos_num(bin.segment_num())
		  && bin.axial_pos_num()<=template_proj_data_info_ptr->get_max_axial_pos_num(bin.segment_num())
		  && bin.segment_num()>=template_proj_data_info_ptr->get_min_segment_num()
		  && bin.segment_num()<=template_proj_data_info_ptr->get_max_segment_num()
		  ) 
		{
		  assert(bin.view_num()>=template_proj_data_info_ptr->get_min_view_num());
		  assert(bin.view_num()<=template_proj_data_info_ptr->get_max_view_num());
            
		  // see if we increment or decrement the value in the sinogram
		  const int event_increment =
		    record.event().is_prompt() 
		    ? ( store_prompts ? 1 : 0 ) // it's a prompt
		    :  delayed_increment;//it is a delayed-coincidence event
            
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
  boost::uniform_int<base_generator_type, unsigned> 
    random_int(generator,
	       0U, total_num_events_in_this_frame-1);
    
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
    
  cerr << "\nFilled in replication vector for " << total_num_events_in_this_frame 
       << " events." << endl;
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
template LmToProjDataBootstrap<LmToProjData>;



END_NAMESPACE_STIR
