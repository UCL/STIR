//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock
  \brief  implementation of the ProjDataRebinning class     
  \author Kris Thielemans
      
  $Date$       
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/

#include "local/stir/recon_buildblock/ProjDataRebinning.h"
#include "stir/Succeeded.h"


START_NAMESPACE_STIR

void 
ProjDataRebinning::
set_defaults()
{
  output_filename_prefix="";
  input_filename="";
  max_segment_num_to_process=-1;
}

void 
ProjDataRebinning::
initialise_keymap()
{
  parser.add_key("input file",&input_filename);
  //parser.add_key("mash x views", &num_views_to_add);
  parser.add_key("maximum absolute segment number to process", &max_segment_num_to_process);

  parser.add_key("output filename prefix",&output_filename_prefix);
 
}


bool 
ProjDataRebinning::
post_processing()
{
  if (output_filename_prefix.length() == 0)// KT 160899 changed name of variable
  { warning("You need to specify an output prefix\n"); return true; }

  if (input_filename.length() == 0)
  { warning("You need to specify an input file\n"); return true; }

#if 0
  if (num_views_to_add!=1 && (num_views_to_add<=0 || num_views_to_add%2 != 0))
  { warning("The 'mash x views' key has an invalid value (must be 1 or even number)\n"); return true; }
#endif
 
  proj_data_sptr= ProjData::read_from_file(input_filename);

  if(max_segment_num_to_process > proj_data_sptr->get_max_segment_num()) 
    { warning("Range error in number of segments\n"); return true;}
  
  return false;
}
#if 0
void 
ProjDataRebinning::
initialise(const string& parameter_filename)
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
      warning("Error parsing input file %s, exiting\n", parameter_filename.c_str());
      exit(EXIT_FAILURE);
    }

  }
}

#endif
 
Succeeded
ProjDataRebinning::
set_up()
{
  return Succeeded::yes;
}

ProjDataRebinning::
~ProjDataRebinning()
{}

END_NAMESPACE_STIR
