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
      
  \date $Date$       
  \version $Revision$
*/

#include "recon_buildblock/ReconstructionParameters.h" 
#include "utilities.h"
#include <iostream>

#ifndef TOMO_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::ends;
#endif

START_NAMESPACE_TOMO


ReconstructionParameters::ReconstructionParameters()
: ParsingObject()
{
}
  
void 
ReconstructionParameters::set_defaults()
{
  input_filename="";
  output_filename_prefix="";
  output_image_size=-1;
  zoom=1.F;
  Xoffset=0.F;
  Yoffset=0.F;
  max_segment_num_to_process=-1;
  num_views_to_add=1;  
  proj_data_ptr=NULL; //MJ added
}


void 
ReconstructionParameters::initialise_keymap()
{
  parser.add_key("input file",&input_filename);
  // KT 03/05/2001 removed
  //parser.add_key("output prefix", &output_filename_prefix);// KT 160899 changed name of variable
  parser.add_key("output filename prefix",&output_filename_prefix);// KT 160899added duplicate key
  parser.add_key("zoom", &zoom);
  // KT 160899 renamed key
  parser.add_key("output image size",&output_image_size);
  parser.add_key("Xoffset (in mm)", &Xoffset);
  parser.add_key("Yoffset (in mm)", &Yoffset);
 
  // KT 180899 new
  parser.add_key("mash x views", &num_views_to_add);

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
  if (zoom < 0)
  { warning("zoom should be positive\n"); return true; }
  // TODO initialise output_image_size from num_bins here or so
  if (output_image_size!=-1 && output_image_size%2==0)
  { warning("output image size needs to be odd\n"); return true; }

  // KT 160899 new
  if (num_views_to_add!=1 && (num_views_to_add<=0 || num_views_to_add%2 != 0))
  { warning("The 'mash x views' key has an invalid value (must be 1 or even number)\n"); return true; }

 
  proj_data_ptr= ProjData::read_from_file(input_filename);
  
  
  return false;
}
 
 


END_NAMESPACE_TOMO




