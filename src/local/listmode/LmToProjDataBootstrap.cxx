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

#include "local/stir/listmode/LmToProjDataBootStrap.h"
#include "stir/Succeeded.h"

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
  parser.add_stop_key("END");
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
    parse(par_filename) ;
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

  if ((store_prompts && delayed_increment!=0) || !store_prompts)
    {
      warning("Can currently only handle prompts or delayed\n");
      return true;
    }

  if (do_time_frame)
    {
      warning("Can currently only handle fixed number of events\n");
      return true;
    }

  if (seed == 0)
    return true;

  // now initialise num_times_to_replicate
  typedef boost::mt19937 base_generator_type;
  base_generator_type generator;

  generator.seed(static_cast<boost::uint32_t>(seed));
  boost::uniform_int<base_generator_type, unsigned> 
    random_int(generator,
	       0U, static_cast<unsigned>(num_events_to_store));// TODO get rid of cast

  num_times_to_replicate.resize(num_events_to_store);

  for (unsigned int i=num_events_to_store; i!=0; --i)
    {
      const unsigned int event_num = random_int();
      num_times_to_replicate[event_num] += 1;
      // warning this did not check for overflow
    }
  num_times_to_replicate_iter = num_times_to_replicate.begin();
  return false;
}


template <typename LmToProjDataT> 
void 
LmToProjDataBootstrap<LmToProjDataT>::get_bin_from_event(Bin& bin, const CListEvent& event) const
{
  assert(num_times_to_replicate_iter != num_times_to_replicate.end());
  if (*num_times_to_replicate_iter > 0)
    {
      base_type::get_bin_from_event(bin, event);
      bin.set_bin_value(bin.get_bin_value() * *num_times_to_replicate_iter);
    }
  ++num_times_to_replicate_iter;
}


// instantiation
template LmToProjDataBootstrap<LmToProjData>;



END_NAMESPACE_STIR
