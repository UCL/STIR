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
#include "stir/is_null_ptr.h"
#include <iostream>
#include "stir/IO/DefaultOutputFileFormat.h"

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
  output_file_format_ptr = new DefaultOutputFileFormat;
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

  parser.add_parsing_key("output file format", &output_file_format_ptr);
 
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
    set_defaults();
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

  // KT 22/05/2003 disabled, as this is done in post_processing
  // proj_data_ptr = ProjData::read_from_file(input_filename_char);

  // KT 28/01/2002 moved here from IterativeReconstruction
  max_segment_num_to_process=
    ask_num("Maximum absolute segment number to process: ",
	    0, proj_data_ptr->get_max_segment_num(), 0);

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
    
    cout << "\n    Enter output file format type :\n";
    output_file_format_ptr = OutputFileFormat::ask_type_and_parameters();

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

  if (is_null_ptr(output_file_format_ptr))
    { warning("output file format has to be set to valid value\n"); return true; }

  // KT 20/06/2001 disabled as not functional yet
#if 0
  if (num_views_to_add!=1 && (num_views_to_add<=0 || num_views_to_add%2 != 0))
  { warning("The 'mash x views' key has an invalid value (must be 1 or even number)\n"); return true; }
#endif
 
  proj_data_ptr= ProjData::read_from_file(input_filename);
  
  
  return false;
}
 
 


END_NAMESPACE_STIR




