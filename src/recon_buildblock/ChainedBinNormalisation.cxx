//
//
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
/*!
  \file
  \ingroup normalisation

  \brief Implementation for class ChainedBinNormalisation

  \author Kris Thielemans
*/


#include "stir/recon_buildblock/ChainedBinNormalisation.h"
#include "stir/is_null_ptr.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

const char * const 
ChainedBinNormalisation::registered_name = "Chained"; 

void 
ChainedBinNormalisation::set_defaults()
{
  apply_first.reset();
  apply_second.reset();
}

void 
ChainedBinNormalisation::
initialise_keymap()
{
  parser.add_start_key("Chained Bin Normalisation Parameters");
  parser.add_parsing_key("Bin Normalisation to apply first", &apply_first);
  parser.add_parsing_key("Bin Normalisation to apply second", &apply_second);
  parser.add_stop_key("END Chained Bin Normalisation Parameters");}

bool 
ChainedBinNormalisation::
post_processing()
{
  return false;
}


ChainedBinNormalisation::
ChainedBinNormalisation()
{
  set_defaults();
}

ChainedBinNormalisation::
ChainedBinNormalisation(shared_ptr<BinNormalisation> const& apply_first_v,
		        shared_ptr<BinNormalisation> const& apply_second_v)
  : apply_first(apply_first_v),
    apply_second(apply_second_v)
{
  post_processing();
}

Succeeded
ChainedBinNormalisation::
set_up(const shared_ptr<ProjDataInfo>& proj_data_info_ptr)
{
  if (!is_null_ptr(apply_first))
    if (apply_first->set_up(proj_data_info_ptr) == Succeeded::no)
      return  Succeeded::no;
  if (!is_null_ptr(apply_second))
    return apply_second->set_up(proj_data_info_ptr);
  else
    return Succeeded::yes;  
}


void 
ChainedBinNormalisation::apply(RelatedViewgrams<float>& viewgrams,const double start_time, const double end_time) const 
{
  if (!is_null_ptr(apply_first))
    apply_first->apply(viewgrams,start_time,end_time);
  if (!is_null_ptr(apply_second))
    apply_second->apply(viewgrams,start_time,end_time);
}

void 
ChainedBinNormalisation::
undo(RelatedViewgrams<float>& viewgrams,const double start_time, const double end_time) const 
{
  if (!is_null_ptr(apply_first))
    apply_first->undo(viewgrams,start_time,end_time);
  if (!is_null_ptr(apply_second))
    apply_second->undo(viewgrams,start_time,end_time);
}

float
ChainedBinNormalisation:: get_bin_efficiency(const Bin& bin,const double start_time, const double end_time) const
{
  return 
    (!is_null_ptr(apply_first) 
     ? apply_first->get_bin_efficiency(bin,start_time,end_time)
     : 1)
    *
    (!is_null_ptr(apply_second) 
     ? apply_second->get_bin_efficiency(bin,start_time,end_time)
     : 1);
}
 
void
ChainedBinNormalisation::
apply(std::vector<Bin>& bins,
           const double start_time, const double end_time) const
{
  if (!is_null_ptr(apply_first))
    apply_first->apply(bins,start_time,end_time);
  if (!is_null_ptr(apply_second))
    apply_second->apply(bins,start_time,end_time);
}

void
ChainedBinNormalisation::
undo(std::vector<Bin>& bins,
           const double start_time, const double end_time) const
{
  if (!is_null_ptr(apply_first))
    apply_first->undo(bins,start_time,end_time);
  if (!is_null_ptr(apply_second))
    apply_second->undo(bins,start_time,end_time);
}
 
std::vector<float>
ChainedBinNormalisation::
get_related_bins_values(const std::vector<Bin>& r_bins) const
{
   error("Not implemented, yet");
   //Something like BINSnormA * BINSnormB
}

END_NAMESPACE_STIR

