//
//
// $Id$
//

/*!

  \file
  \ingroup recon_buildblock
  
  \brief  implementation of the ReconstructionParameters class 
    
  \author Matthew Jacobson
  \author Kris Thielemans
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

// KT&CL 160899 added arguments

ReconstructionParameters::ReconstructionParameters(): KeyParser()
{
  // remove set-up of default values
  //MJ 25/03/2000 put them back

  input_filename="";
  output_filename_prefix="";
  output_image_size=-1;
  zoom=1.F;
  Xoffset=0.F;
  Yoffset=0.F;
  max_segment_num_to_process=-1;
  num_views_to_add=1;
  disp=0;
  save_intermediate_files=1;

  
  proj_data_ptr=NULL; //MJ added

  add_key("input file", 
    KeyArgument::ASCII, &input_filename);
  // TODO remove next key
  add_key("output prefix", 
    KeyArgument::ASCII, &output_filename_prefix);// KT 160899 changed name of variable
  add_key("output filename prefix", 
    KeyArgument::ASCII, &output_filename_prefix);// KT 160899added duplicate key
 
  add_key("display (0,1,2)",
    KeyArgument::INT, &disp);
  // KT 25/05/2000 Save -> save
  add_key("save intermediate files",
    KeyArgument::INT, &save_intermediate_files);
  add_key("zoom",         
    KeyArgument::DOUBLE, &zoom);
  // KT 160899 renamed key
  add_key("output image size",
    KeyArgument::INT, &output_image_size);
  add_key("Xoffset (in mm)",
    KeyArgument::DOUBLE, &Xoffset);
  add_key("Yoffset (in mm)",          
    KeyArgument::DOUBLE, &Yoffset);
 
  // KT 180899 new
  add_key("mash x views",
    KeyArgument::INT, &num_views_to_add);

  add_key("maximum absolute segment number to process",
    KeyArgument::INT, &max_segment_num_to_process);
 
  add_key("END", 
    KeyArgument::NONE, &KeyParser::stop_parsing);
 
}

void 
ReconstructionParameters::initialise(const string& parameter_filename)
{
  if(parameter_filename.size()==0)
  {
    cerr << "Next time, try passing the executable a parameter file"
	 << endl;

    ask_parameters();
  }

else
  {
  if (!parse(parameter_filename.c_str()))
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
 
 

//CL 01/08/99 Change into string
// KT 160899 added const to method
string ReconstructionParameters::parameter_info() const
{
    char str[10000];
    ostrstream s(str, 10000);
    

    //MJ 01/02/2000 Got rid of annoying spaces
    s << "input file := " << input_filename  << endl;
    // KT 25/05/2000 file -> filename
    s << "output filename prefix := " << output_filename_prefix << endl;
// KT 160899 changed name of variable
    s << "display (0,1,2) := " << disp << endl;
    s << "save intermediate files := " << save_intermediate_files << endl;
    s << "zoom := " << zoom << endl;
    s << "Xoffset (in mm) := " << Xoffset << endl;
    s << "Yoffset (in mm) := " << Yoffset << endl;
    s << "mash x views := " << num_views_to_add << endl;
    s << "output image size := " << output_image_size  << endl;
    s << "maximum absolute segment number to process := "
      << max_segment_num_to_process << endl<<endl;
    s << ends;
    
    return s.str();    
}


END_NAMESPACE_TOMO




