//
//
/*!

  \file
  \ingroup listmode
  \brief Class stir::LmToProjDataWithRandomRejection for rebinning listmode files rejection some events randomly
    
  \author Kris Thielemans
      
*/
/*
    Copyright (C) 2003- 2012, Hammersmith Imanet Ltd
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

#include "local/stir/listmode/LmToProjDataWithRandomRejection.h"
#include "stir/listmode/CListRecord.h"
#include "stir/Succeeded.h"
#include <iostream>
#include <algorithm>



START_NAMESPACE_STIR

template <typename LmToProjDataT>
void 
LmToProjDataWithRandomRejection<LmToProjDataT>::set_defaults()
{
  LmToProjData::set_defaults();
  this->seed = 42;
  this->reject_if_above = .5F;
}

template <typename LmToProjDataT>
void 
LmToProjDataWithRandomRejection<LmToProjDataT>::initialise_keymap()
{
  LmToProjData::initialise_keymap();
  this->parser.add_start_key("LmToProjDataWithRandomRejection Parameters");
  this->parser.add_key("seed", reinterpret_cast<int *>(&seed)); // TODO get rid of cast
  this->parser.add_key("reject_if_above", &reject_if_above);
}

template <typename LmToProjDataT>
LmToProjDataWithRandomRejection<LmToProjDataT>::
LmToProjDataWithRandomRejection(const char * const par_filename)
{
  set_defaults();
  if (par_filename!=0)
    this->parse(par_filename) ;
  else
    this->ask_parameters();
}

template <typename LmToProjDataT>
LmToProjDataWithRandomRejection<LmToProjDataT>::
LmToProjDataWithRandomRejection(const char * const par_filename, const unsigned int seed_v)
{
  set_defaults();
  seed = seed_v;
  if (par_filename!=0)
    {
      this->parse(par_filename);
      // make sure that seed_v parameter overrides whatever was in the par file
      if (seed != seed_v)
	{
	  warning("LmToProjDataWithRandomRejection: parameter file %s contains seed (%u) which is\n"
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
LmToProjDataWithRandomRejection<LmToProjDataT>::
post_processing()
{
  if (LmToProjData::post_processing())
    return true;

  if (this->seed == 0)
    {
      warning("Seed needs to be non-zero"); return true;
    }

  if (this->reject_if_above<0.F || this->reject_if_above>1.F)
    {
      warning("reject_if_above needs to be between 0 and 1"); return true;
    }

  return false;
}

template <typename LmToProjDataT> 
float
LmToProjDataWithRandomRejection<LmToProjDataT>::
set_reject_if_above(const float v)
{
  const float ret = this->reject_if_above;
  this->reject_if_above = v;
  return ret;
}


 

template <typename LmToProjDataT> 
void 
LmToProjDataWithRandomRejection<LmToProjDataT>::
start_new_time_frame(const unsigned int new_frame_num)
{

  base_type::start_new_time_frame(new_frame_num);
  this->random_generator.seed(static_cast<boost::uint32_t>(seed));
}

template <typename LmToProjDataT> 
void 
LmToProjDataWithRandomRejection<LmToProjDataT>::
get_bin_from_event(Bin& bin, const CListEvent& event) const
{

  static boost::uniform_01<random_generator_type> random01(random_generator);

  const double randnum=random01();
  //std::cout << randnum << '\n';
  if (randnum <= this->reject_if_above)
    {
      base_type::get_bin_from_event(bin, event);
    }
  else
    bin.set_bin_value(-1);
}


// instantiation
template class LmToProjDataWithRandomRejection<LmToProjData>;



END_NAMESPACE_STIR
