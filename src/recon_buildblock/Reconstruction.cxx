//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2011-12-31, Hammersmith Imanet Ltd

    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/

/*!

  \file
  \ingroup recon_buildblock
  
  \brief  implementation of the stir::Reconstruction class 
    
  \author Kris Thielemans
  \author PARAPET project
      
*/

#include "stir/recon_buildblock/Reconstruction.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/DiscretisedDensity.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"
#include "stir/info.h"


#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/modelling/KineticParameters.h"

START_NAMESPACE_STIR

template <typename TargetT>
Reconstruction<TargetT>::Reconstruction()
{
  this->set_defaults();
}


// parameters

template <typename TargetT>
void 
Reconstruction<TargetT>::set_defaults()
{
  this->_already_set_up=false;
  this->output_filename_prefix="";
  this->output_file_format_ptr =
    OutputFileFormat<TargetT>::default_sptr();
  this->post_filter_sptr.reset();

  this->_disable_output = false;
  this->_verbosity = -1;

}

template <typename TargetT>
void 
Reconstruction<TargetT>::initialise_keymap()
{

  this->parser.add_key("output filename prefix",&this->output_filename_prefix);
  this->parser.add_parsing_key("output file format type", &this->output_file_format_ptr);
  this->parser.add_parsing_key("post-filter type", &this->post_filter_sptr); 
  this->parser.add_key("disable output", &_disable_output);
  this->parser.add_key("verbosity", &_verbosity);
//  parser.add_key("END", &KeyParser::stop_parsing);
 
}

template <typename TargetT>
void 
Reconstruction<TargetT>::initialise(const std::string& parameter_filename)
{
  _already_set_up = false;
  if(parameter_filename.size()==0)
  {
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

  if ((this->_disable_output) & (this->get_registered_name ()=="KOSMAPOSL") )
  { warning("You have disabled the alpha coefficient output. Only emission image files will be written to "
            "disk after or during reconstuction"); }

  else if (this->_disable_output)
  { warning("You have disabled the output. No files will be written to "
            "disk after or during reconstuction"); }

  if (this->output_filename_prefix.length() == 0 &&
          !this->_disable_output)// KT 160899 changed name of variable
  { warning("You need to specify an output prefix"); return true; }

  if (is_null_ptr(this->output_file_format_ptr))
    { warning("output file format has to be set to valid value"); return true; }

  if (_verbosity >= 0)
      Verbosity::set(_verbosity);

  return false;
}

template <typename TargetT>
void
Reconstruction<TargetT>::
set_output_filename_prefix(const std::string& arg)
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
  _already_set_up = false;
  this->post_filter_sptr  = arg;
}
 
template <typename TargetT>
Succeeded
Reconstruction<TargetT>::
set_up(shared_ptr<TargetT> const& target_data_sptr_v)
{
  _already_set_up = true;
  this->target_data_sptr = target_data_sptr_v;

  if(!is_null_ptr(this->post_filter_sptr)) 
  {
    info("Building post filter kernel");
    
    if (this->post_filter_sptr->set_up(*target_data_sptr)
          == Succeeded::no)
      {
	warning("Error building post filter");
	return Succeeded::no;
      }
  }
  return Succeeded::yes;
}

template <typename TargetT>
void
Reconstruction<TargetT>::
check(TargetT const& target_data) const
{
  if (!this->_already_set_up)
    error("Reconstruction method called without calling set_up first.");
  if (! this->target_data_sptr->has_same_characteristics(target_data))
    error("Reconstruction set-up with different geometry for target.");
}

template <typename TargetT>
void
Reconstruction<TargetT>::
set_disable_output(bool _val)
{
    this->_disable_output = _val;
}

template < typename TargetT>
shared_ptr<TargetT >
Reconstruction<TargetT>::
get_target_image()
{
    return target_data_sptr;
}

template class Reconstruction<DiscretisedDensity<3,float> >; 
template class Reconstruction<ParametricVoxelsOnCartesianGrid >; 
END_NAMESPACE_STIR

