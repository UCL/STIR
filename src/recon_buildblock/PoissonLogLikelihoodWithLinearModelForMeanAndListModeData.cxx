//
// $Id$ 
// 
/* 
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
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
 
  $Date$ 
  $Revision$ 
*/ 
 
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndListModeData.h" 
#include "stir/VoxelsOnCartesianGrid.h" 
#include "stir/Succeeded.h" 

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
  this->list_mode_data_sptr = NULL; 
  this->current_frame_num =0; 
 
  this->output_image_size_xy=-1; 
  this->output_image_size_z=-1; 
  this->zoom=1.F; 
  this->Xoffset=0.F; 
  this->Yoffset=0.F; 
  this->Zoffset=0.F; 
 
} 

template <typename TargetT>  
void  
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>::
initialise_keymap() 
{ 
  base_type::initialise_keymap(); 
  this->parser.add_key("list mode filename", &this->list_mode_filename); 
  this->parser.add_key("zoom", &this->zoom); 
  this->parser.add_key("XY output image size (in pixels)",&this->output_image_size_xy); 
  this->parser.add_key("Z output image size (in pixels)",&this->output_image_size_z); 
  //parser.add_key("X offset (in mm)", &Xoffset); // KT 10122001 added spaces 
  //parser.add_key("Y offset (in mm)", &Yoffset); 
   
  //this->parser.add_key("Z offset (in mm)", &Zoffset); 
  this->parser.add_key("time frame definition filename", &this->frame_defs_filename);
  // SM TODO -- later do not parse
  this->parser.add_key("time frame number", &this->current_frame_num);
     
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
   CListModeData::read_from_file(this->list_mode_filename); 

  if (this->frame_defs_filename.size()!=0)
    this->frame_defs = TimeFrameDefinitions(this->frame_defs_filename);
  else
    {
      // make a single frame starting from 0. End value will be ignored.
      vector<pair<double, double> > frame_times(1, pair<double,double>(0,1));
      this->frame_defs = TimeFrameDefinitions(frame_times);
    } 
  // image stuff 
  if (this->zoom <= 0) 
  { warning("zoom should be positive\n"); return true; } 
   
  if (this->output_image_size_xy!=-1 && this->output_image_size_xy<1) // KT 10122001 appended_xy 
  { warning("output image size xy must be positive (or -1 as default)\n"); return true; } 
  if (this->output_image_size_z!=-1 && this->output_image_size_z<1) // KT 10122001 new 
  { warning("output image size z must be positive (or -1 as default)\n"); return true; } 
   

   return false; 
} 

#if 0
Succeeded  
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>::
set_up(shared_ptr <TargetT > const& target_sptr) 
{ 
  if ( base_type::set_up(target_sptr) != Succeeded::yes) 
    return Succeeded::no; 
 
    return Succeeded::yes; 
 
 
} 
 
template <typename TargetT> 
TargetT * 
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>:: 
construct_target_ptr() const 
{ 
  return 
      new VoxelsOnCartesianGrid<float> (*this->proj_data_sptr->get_proj_data_info_ptr(), 
					this->zoom, 
					CartesianCoordinate3D<float>(this->Zoffset, 
								     this->Yoffset, 
								     this->Xoffset), 
					CartesianCoordinate3D<int>(this->output_image_size_z, 
                                                                   this->output_image_size_xy, 
                                                                   this->output_image_size_xy) 
                                       );
} 
#endif

#  ifdef _MSC_VER
// prevent warning message on instantiation of abstract class 
#  pragma warning(disable:4661)
#  endif

template class 
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<DiscretisedDensity<3,float> >;



END_NAMESPACE_STIR
