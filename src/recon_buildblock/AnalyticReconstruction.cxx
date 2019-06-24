/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2006,  Hammersmith Imanet Ltd 
    Copyright (C) 2016, 2018 - 2019 University College London
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
  \ingroup recon_buildblock
  
  \brief  implementation of the stir::AnalyticReconstruction class 
    
  \author Kris Thielemans
  \author Matthew Jacobson
  \author Nikos Efthimiou
  \author PARAPET project
      
*/

#include "stir/recon_buildblock/AnalyticReconstruction.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"
#include "stir/info.h"
#include <iostream>
#include "stir/IO/OutputFileFormat.h"

START_NAMESPACE_STIR



// parameters

void 
AnalyticReconstruction::set_defaults()
{
  base_type::set_defaults();
  input_filename="";
  max_segment_num_to_process=-1;
  proj_data_ptr.reset(); 
  target_parameter_parser.set_defaults();
}


void 
AnalyticReconstruction::initialise_keymap()
{
  base_type::initialise_keymap();
  parser.add_key("input file",&input_filename);
  // KT 20/06/2001 disabled
  //parser.add_key("mash x views", &num_views_to_add);

  parser.add_key("maximum absolute segment number to process", &max_segment_num_to_process);

  this->target_parameter_parser.add_to_keymap(parser);

//  parser.add_key("END", &KeyParser::stop_parsing);
 
}

#if 0
// disable ask_parameters (just use default version)
void AnalyticReconstruction::ask_parameters()
{

  char output_filename_prefix_char[max_filename_length];
  char input_filename_char[max_filename_length];
  ask_filename_with_extension(input_filename_char,"Enter file name of 3D sinogram data : ", ".hs");
  cerr<<endl;

  input_filename=input_filename_char;

  // KT 22/05/2003 disabled, as this is done in post_processing
  // proj_data_ptr = ProjData::read_from_file(input_filename_char);

  // KT 28/01/2002 moved here from IterativeAnalyticReconstruction
  max_segment_num_to_process=
    ask_num("Maximum absolute segment number to process: ",
	    0, proj_data_ptr->get_max_segment_num(), 0);
#if 0
    // The angular compression consists of an average pairs of sinograms rows
    // in order to reduce the number of views by a factor of 2
    // and therefore reduce the amount of data in a sinogram as well
    // the reconstruction time by about the half of
    // the total unmashed recontruction time
    // By default, no mashing
    // Note: The inclusion of angular compression has been shown in literature
    // to have little effect near the center of the FOV.
    // However, it could cause loss of precision
    num_views_to_add=  ask_num("Mashing views ? (1: No mashing, 2: By 2 , 4: By 4) : ",1,4,1);
#endif

  ask_filename_with_extension(output_filename_prefix_char,"Output filename prefix", "");

  output_filename_prefix=output_filename_prefix_char;

}
#endif // ask_parameters disabled


bool AnalyticReconstruction::post_processing()
{
  if (base_type::post_processing()) 
    return true; 
  if (input_filename.length() == 0)
  { warning("You need to specify an input file\n"); return true; }
  // KT 20/06/2001 disabled as not functional yet
#if 0
  if (num_views_to_add!=1 && (num_views_to_add<=0 || num_views_to_add%2 != 0))
  { warning("The 'mash x views' key has an invalid value (must be 1 or even number)\n"); return true; }
#endif
 
  proj_data_ptr= ProjData::read_from_file(input_filename);

  target_parameter_parser.check_values();

  return false;
}


//************* other functions *************

DiscretisedDensity<3,float>* 
AnalyticReconstruction::
construct_target_image_ptr() const
{
  return
    this->target_parameter_parser.create(this->get_input_data());
}



Succeeded 
AnalyticReconstruction::
reconstruct() 
{
  shared_ptr<DiscretisedDensity<3,float> > target_image_ptr(construct_target_image_ptr());
  const Succeeded success = this->reconstruct(target_image_ptr);
  if (success == Succeeded::yes && !_disable_output)
  {
    this->output_file_format_ptr->
      write_to_file(this->output_filename_prefix, *target_image_ptr);
  }
  return success;
}
 

Succeeded 
AnalyticReconstruction::
reconstruct(shared_ptr<TargetT> const& target_image_sptr)
{
  const Succeeded success = this->actual_reconstruct(target_image_sptr);
  if (success == Succeeded::yes)
  {
    if(!is_null_ptr(this->post_filter_sptr))
      {
	info("Applying post-filter");
	this->post_filter_sptr->apply(*target_image_sptr);
	
	info(boost::format("  min and max after post-filtering %1% %2%") % target_image_sptr->find_min() % target_image_sptr->find_max());
      }
  }
  return success;
}

void
AnalyticReconstruction::
set_input_data(const shared_ptr<ExamData> &arg)
{
	this->proj_data_ptr = dynamic_pointer_cast < ProjData >(arg);
}

const ProjData&
AnalyticReconstruction::
get_input_data() const
{
  return *this->proj_data_ptr;
}

// forwarding functions for ParseDiscretisedDensityParameters
int
AnalyticReconstruction::
get_output_image_size_xy() const
{ return target_parameter_parser.get_output_image_size_xy(); }

void
AnalyticReconstruction::
set_output_image_size_xy(int v)
{ target_parameter_parser.set_output_image_size_xy(v); }

int
AnalyticReconstruction::
get_output_image_size_z() const
{ return target_parameter_parser.get_output_image_size_z(); }

void
AnalyticReconstruction::
set_output_image_size_z(int v)
{ target_parameter_parser.set_output_image_size_z(v); }

float
AnalyticReconstruction::
get_zoom_xy() const
{ return target_parameter_parser.get_zoom_xy(); }

void
AnalyticReconstruction::
set_zoom_xy(float v)
{ target_parameter_parser.set_zoom_xy(v); }

float
AnalyticReconstruction::
get_zoom_z() const
{ return target_parameter_parser.get_zoom_z(); }

void
AnalyticReconstruction::
set_zoom_z(float v)
{ target_parameter_parser.set_zoom_z(v); }

const CartesianCoordinate3D<float>&
AnalyticReconstruction::
get_offset() const
{ return target_parameter_parser.get_offset(); }

void
AnalyticReconstruction::
set_offset(const CartesianCoordinate3D<float>& v)
{ target_parameter_parser.set_offset(v); }

END_NAMESPACE_STIR

