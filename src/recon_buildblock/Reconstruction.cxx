//
// $Id$
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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
  
  \brief  implementation of the stir::Reconstruction class 
    
  \author Kris Thielemans
  \author PARAPET project
      
  $Date$       
  $Revision$
*/

#include "stir/recon_buildblock/Reconstruction.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/DiscretisedDensity.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"
#include <iostream>

#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/modelling/KineticParameters.h"

START_NAMESPACE_STIR



// parameters

template <typename TargetT>
void 
Reconstruction<TargetT>::set_defaults()
{
  this->output_filename_prefix="";
  this->output_file_format_ptr =
    OutputFileFormat<TargetT>::default_sptr();
  this->post_filter_sptr = 0;

}

template <typename TargetT>
void 
Reconstruction<TargetT>::initialise_keymap()
{

  this->parser.add_key("output filename prefix",&this->output_filename_prefix);
  this->parser.add_parsing_key("output file format type", &this->output_file_format_ptr);
  this->parser.add_parsing_key("post-filter type", &this->post_filter_sptr); 
 
//  parser.add_key("END", &KeyParser::stop_parsing);
 
}

template <typename TargetT>
void 
Reconstruction<TargetT>::initialise(const string& parameter_filename)
{
  if(parameter_filename.size()==0)
  {
    cerr << "Next time, try passing the executable a parameter file"
	 << endl;

    this->set_defaults();
    this->ask_parameters();
  }

else
  {
    this->set_defaults();
    if(!this->parse(parameter_filename.c_str()))
    {
      error("Error parsing input file %s, exiting", parameter_filename.c_str());
    }

  }
}


template <typename TargetT>
bool 
Reconstruction<TargetT>::
post_processing()
{
  if (this->output_filename_prefix.length() == 0)// KT 160899 changed name of variable
  { warning("You need to specify an output prefix"); return true; }

  if (is_null_ptr(this->output_file_format_ptr))
    { warning("output file format has to be set to valid value"); return true; }
  
  return false;
}

template <typename TargetT>
void
Reconstruction<TargetT>::
set_output_filename_prefix(const string& arg)
{
  this->output_filename_prefix  = arg;
}

template <typename TargetT>
void
Reconstruction<TargetT>::
set_output_file_format_ptr(const shared_ptr<OutputFileFormat<TargetT> >& arg)
{
  this->output_file_format_ptr  = arg;
}

template <typename TargetT>
void
Reconstruction<TargetT>::
set_post_processor_sptr(const shared_ptr<DataProcessor<TargetT> > & arg)
{
  this->post_filter_sptr  = arg;
}
 
template <typename TargetT>
Succeeded
Reconstruction<TargetT>::
set_up(shared_ptr<TargetT> const& target_data_sptr)
{

  if(!is_null_ptr(this->post_filter_sptr)) 
  {
    cerr<<endl<<"Building post filter kernel"<<endl;
    
    if (this->post_filter_sptr->set_up(*target_data_sptr)
          == Succeeded::no)
      {
	warning("Error building post filter");
	return Succeeded::no;
      }
  }
  return Succeeded::yes;
}

template class Reconstruction<DiscretisedDensity<3,float> >; 
template class Reconstruction<ParametricVoxelsOnCartesianGrid >; 
END_NAMESPACE_STIR

