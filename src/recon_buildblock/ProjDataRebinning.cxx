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

#include "stir/recon_buildblock/ProjDataRebinning.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"

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
  if (output_filename_prefix.length() == 0)
  { warning("You need to specify an output prefix\n"); return true; }

  if (input_filename.length() == 0)
  { warning("You need to specify an input file\n"); return true; }

#if 0
  if (num_views_to_add!=1 && (num_views_to_add<=0 || num_views_to_add%2 != 0))
  { warning("The 'mash x views' key has an invalid value (must be 1 or even number)\n"); return true; }
#endif
 
  proj_data_sptr= ProjData::read_from_file(input_filename);
  
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
  if (is_null_ptr(proj_data_sptr))
    {
      warning("ProjDataRebinning: input projection data not set");
      return Succeeded::no;
    } 
  if (max_segment_num_to_process == -1)
    max_segment_num_to_process = proj_data_sptr->get_max_segment_num();
  else  if(max_segment_num_to_process > proj_data_sptr->get_max_segment_num()) 
    { 
      warning("ProjDataRebinning: Range error in number of segments to process.\n"
	      "Max segment number in data is %d while you asked for %d",
	      proj_data_sptr->get_max_segment_num(),
	      max_segment_num_to_process); 
      return Succeeded::no;
    }
  return Succeeded::yes;
}

ProjDataRebinning::
~ProjDataRebinning()
{}

void ProjDataRebinning::set_max_segment_num_to_process(int ns)
{ max_segment_num_to_process = ns;
}

int ProjDataRebinning::get_max_segment_num_to_process() const
{ return max_segment_num_to_process;
}
void 
ProjDataRebinning::set_output_filename_prefix(const string& s)
{
  output_filename_prefix = s;
}
string 
ProjDataRebinning::get_output_filename_prefix() const
{
  return output_filename_prefix;
}


shared_ptr<ProjData> 
ProjDataRebinning::
get_proj_data_sptr()
{
  /* KT: deleted warning messages about null pointers. 
     The user should check this, and might not want the have the 
     warnings written to stderr. */
     return proj_data_sptr;
 } 


void 
ProjDataRebinning::
set_input_proj_data_sptr(const shared_ptr<ProjData>& new_proj_data_sptr)
{
  proj_data_sptr = new_proj_data_sptr;
} 


END_NAMESPACE_STIR
