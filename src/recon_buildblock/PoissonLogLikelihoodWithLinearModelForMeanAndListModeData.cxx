//
// 
/* 
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
    Copyright (C) 2018, University College London
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
  \ingroup GeneralisedObjectiveFunction 
  \brief Declaration of class
  stir::PoissonLogLikelihoodWithLinearModelForMeanAndListModeData 
 
  \author Kris Thielemans 
  \author Sanida Mustafovic 
 
*/ 
 
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndListModeData.h" 
#include "stir/VoxelsOnCartesianGrid.h" 
#include "stir/Succeeded.h" 
#include "stir/IO/read_from_file.h"

using std::vector;
using std::pair;

START_NAMESPACE_STIR

 
template <typename TargetT>    
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>:: 
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData() 
{ 
  this->set_defaults(); 
} 

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>::
set_defaults() 
{ 
  base_type::set_defaults(); 
  this->list_mode_filename =""; 
  this->frame_defs_filename ="";
  this->list_mode_data_sptr.reset(); 
  this->current_frame_num = 1;
  this->num_events_to_use = 0L;
 
  this->target_parameter_parser.set_defaults();
 
} 

template <typename TargetT>  
void  
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>::
initialise_keymap() 
{ 
  base_type::initialise_keymap(); 
  this->parser.add_key("list mode filename", &this->list_mode_filename); 
  this->target_parameter_parser.add_to_keymap(this->parser);
  this->parser.add_key("time frame definition filename", &this->frame_defs_filename);
  // SM TODO -- later do not parse
  this->parser.add_key("time frame number", &this->current_frame_num);
       this->parser.add_parsing_key("Bin Normalisation type", &this->normalisation_sptr);
} 

template <typename TargetT>     
bool  
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>::post_processing() 
{ 
  if (base_type::post_processing() == true) 
  return true; 

  if (this->list_mode_filename.length() == 0) 
  { warning("You need to specify an input file\n"); return true; } 

  this->list_mode_data_sptr=
    read_from_file<CListModeData>(this->list_mode_filename); 

  if (this->frame_defs_filename.size()!=0)
    this->frame_defs = TimeFrameDefinitions(this->frame_defs_filename);
  else
    {
      // make a single frame starting from 0. End value will be ignored.
      vector<pair<double, double> > frame_times(1, pair<double,double>(0,0));
      this->frame_defs = TimeFrameDefinitions(frame_times);
    } 
  target_parameter_parser.check_values();

  return false;
} 

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>::
set_input_data(const shared_ptr<ExamData> & arg)
{
    this->list_mode_data_sptr = dynamic_pointer_cast<CListModeData>(arg);
}

template <typename TargetT>
const CListModeData&
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>::
get_input_data() const
{
  return *this->list_mode_data_sptr;
}

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>::
set_additive_proj_data_sptr(const shared_ptr<ExamData> &arg)
{
    this->additive_proj_data_sptr = dynamic_pointer_cast<ProjData>(arg);
}

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>::
set_normalisation_sptr(const shared_ptr<BinNormalisation>& arg)
{
  this->normalisation_sptr = arg;
}

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>::
start_new_time_frame(const unsigned int)
{}

template<typename TargetT>
Succeeded
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>::
set_up(shared_ptr <TargetT > const& target_sptr)
{
  if ( base_type::set_up(target_sptr) != Succeeded::yes)
    return Succeeded::no;

  // handle time frame definitions etc
    if(this->num_events_to_use==0 && this->frame_defs_filename.size() == 0)
      do_time_frame = true;
 
    return Succeeded::yes;
}

#  ifdef _MSC_VER
// prevent warning message on instantiation of abstract class 
#  pragma warning(disable:4661)
#  endif

template class 
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<DiscretisedDensity<3,float> >;



END_NAMESPACE_STIR
