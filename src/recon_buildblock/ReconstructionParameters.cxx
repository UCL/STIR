//
// $Id$
//

/*!

  \file
  \ingroup recon_buildblock
  
  \brief  implementation of the ReconstructionParameters class 
    
  \author Matthew Jacobson
  \author Kris Thielemans
  \author Sanida Mustafovic
  \author PARAPET project
      
  $Date$       
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/recon_buildblock/ReconstructionParameters.h" 
#include "stir/utilities.h"
#include <iostream>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::ends;
#endif

START_NAMESPACE_STIR


ReconstructionParameters::ReconstructionParameters()
: ParsingObject()
{
}
  
void 
ReconstructionParameters::set_defaults()
{
  input_filename="";
  output_filename_prefix="";
  output_image_size_xy=-1;
  output_image_size_z=-1;
  zoom=1.F;
  Xoffset=0.F;
  Yoffset=0.F;
  // KT 20/06/2001 new
  Zoffset=0.F;
  max_segment_num_to_process=-1;
  // KT 20/06/2001 disabled
  //num_views_to_add=1;  
  proj_data_ptr=NULL; //MJ added
}


void 
ReconstructionParameters::initialise_keymap()
{
  parser.add_key("input file",&input_filename);
  // KT 03/05/2001 removed
  //parser.add_key("output prefix", &output_filename_prefix);
  parser.add_key("output filename prefix",&output_filename_prefix);
  parser.add_key("zoom", &zoom);
  // KT 10122001 renamed key and added z version
  parser.add_key("XY output image size (in pixels)",&output_image_size_xy);
  // KT 13122001 disabled Z stuff and offsets as it still gives problems with the projectors
  // parser.add_key("Z output image size (in pixels)",&output_image_size_z);
  //parser.add_key("X offset (in mm)", &Xoffset); // KT 10122001 added spaces
  //parser.add_key("Y offset (in mm)", &Yoffset);
  // KT 20/06/2001 new
  // parser.add_key("Z offset (in mm)", &Zoffset);
 
  // KT 20/06/2001 disabled
  //parser.add_key("mash x views", &num_views_to_add);

  parser.add_key("maximum absolute segment number to process", &max_segment_num_to_process);
 
//  parser.add_key("END", &KeyParser::stop_parsing);
 
}

void 
ReconstructionParameters::initialise(const string& parameter_filename)
{
  if(parameter_filename.size()==0)
  {
    cerr << "Next time, try passing the executable a parameter file"
	 << endl;

    set_defaults();
    ask_parameters();
  }

else
  {
  if(!parse(parameter_filename.c_str()))
    {
      error("Error parsing input file %s, exiting\n", parameter_filename.c_str());
    }

  }
}

//MJ new
void ReconstructionParameters::ask_parameters()
{

  char input_filename_char[max_filename_length];
  char output_filename_prefix_char[max_filename_length];

  ask_filename_with_extension(input_filename_char,"Enter file name of 3D sinogram data : ", ".hs");
  cerr<<endl;

  input_filename=input_filename_char;

  proj_data_ptr = ProjData::read_from_file(input_filename_char);

  ask_filename_with_extension(output_filename_prefix_char,"Output filename prefix", "");

  output_filename_prefix=output_filename_prefix_char;


}



bool ReconstructionParameters::post_processing()
{
  if (input_filename.length() == 0)
  { warning("You need to specify an input file\n"); return true; }
  if (output_filename_prefix.length() == 0)// KT 160899 changed name of variable
  { warning("You need to specify an output prefix\n"); return true; }
  if (zoom <= 0)
  { warning("zoom should be positive\n"); return true; }
  
  if (output_image_size_xy!=-1 && output_image_size_xy<1) // KT 10122001 appended_xy
  { warning("output image size xy must be positive (or -1 as default)\n"); return true; }
  if (output_image_size_z!=-1 && output_image_size_z<1) // KT 10122001 new
  { warning("output image size z must be positive (or -1 as default)\n"); return true; }


  // KT 20/06/2001 disabled as not functional yet
#if 0
  if (num_views_to_add!=1 && (num_views_to_add<=0 || num_views_to_add%2 != 0))
  { warning("The 'mash x views' key has an invalid value (must be 1 or even number)\n"); return true; }
#endif
 
  proj_data_ptr= ProjData::read_from_file(input_filename);
  
  
  return false;
}
 
 


END_NAMESPACE_STIR




