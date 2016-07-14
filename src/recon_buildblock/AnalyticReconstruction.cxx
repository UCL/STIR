/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2006,  Hammersmith Imanet Ltd 
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
    
  \author Matthew Jacobson
  \author Kris Thielemans
  \author Sanida Mustafovic
  \author PARAPET project
      
*/

#include "stir/recon_buildblock/AnalyticReconstruction.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/Succeeded.h"
#include "stir/utilities.h"
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
  output_image_size_xy=-1;
  output_image_size_z=-1;
  zoom=1.F;
  Xoffset=0.F;
  Yoffset=0.F;
  Zoffset=0.F;
}


void 
AnalyticReconstruction::initialise_keymap()
{
  base_type::initialise_keymap();
  parser.add_key("input file",&input_filename);
  // KT 20/06/2001 disabled
  //parser.add_key("mash x views", &num_views_to_add);

  parser.add_key("maximum absolute segment number to process", &max_segment_num_to_process);

  parser.add_key("zoom", &zoom);
  parser.add_key("XY output image size (in pixels)",&output_image_size_xy);
  parser.add_key("Z output image size (in pixels)",&output_image_size_z);
  //parser.add_key("X offset (in mm)", &Xoffset); // KT 10122001 added spaces
  //parser.add_key("Y offset (in mm)", &Yoffset);
  
  parser.add_key("Z offset (in mm)", &Zoffset);

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

  zoom=  ask_num("Specify a zoom factor as magnification effect ? ",0.1,10.,1.);


  output_image_size_xy =  
    ask_num("Final image size (-1 for default)? ",
	    -1,
	    4*static_cast<int>(proj_data_ptr->get_num_tangential_poss()*zoom),
	    -1);
    
#if 0    
    // This section enables you to position a reconstructed image
    // along x (horizontal), y (vertical) and/or z (transverse) axes
    // The default values is in the center of the FOV,
    // the positve direction is
    // for x-axis, toward the patient's left side (assuming typical spinal, head first position)
    // for y-axis, toward the top of the FOV
    // for z-axis, toward the patient's feet (assuming typical spinal, head first position)
    
    cout << endl << "    Enter offset  Xoff, Yoff (in pixels) :";
    Xoffset = ask_num("   X offset  ",-old_size/2, old_size/2, 0);
    Yoffset = ask_num("   Y offset  ",-old_size/2, old_size/2, 0);
#endif

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

  if (zoom <= 0)
  { warning("zoom should be positive\n"); return true; }
  
  if (output_image_size_xy!=-1 && output_image_size_xy<1) // KT 10122001 appended_xy
  { warning("output image size xy must be positive (or -1 as default)\n"); return true; }
  if (output_image_size_z!=-1 && output_image_size_z<1) // KT 10122001 new
  { warning("output image size z must be positive (or -1 as default)\n"); return true; }

  
  return false;
}


//************* other functions *************

DiscretisedDensity<3,float>* 
AnalyticReconstruction::
construct_target_image_ptr() const
{
  return
      new VoxelsOnCartesianGrid<float> (*this->proj_data_ptr->get_proj_data_info_ptr(),
					static_cast<float>(this->zoom),
					CartesianCoordinate3D<float>(static_cast<float>(this->Zoffset),
								     static_cast<float>(this->Yoffset),
								     static_cast<float>(this->Xoffset)),
					CartesianCoordinate3D<int>(this->output_image_size_z,
                                                                   this->output_image_size_xy,
                                                                   this->output_image_size_xy)
                                       );
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
    this->proj_data_ptr.reset(dynamic_cast < ProjData * > (arg.get()) );
}

void
AnalyticReconstruction::
set_additive_proj_data_sptr(const shared_ptr<ExamData> &arg)
{
    error("Not implemented yet");
}

void
AnalyticReconstruction::
set_normalisation_sptr(const shared_ptr<BinNormalisation>& arg)
{
    error("Not implemented yet");
}
 
END_NAMESPACE_STIR

