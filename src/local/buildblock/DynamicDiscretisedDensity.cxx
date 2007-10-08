//
// $Id$
//
/*!
  \file
  \ingroup utilities
  \brief Implementation of class stir::DynamicDiscretisedDensity
  \author Kris Thielemans
  \author Charalampos Tsoumpas
  
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
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

#include "local/stir/DynamicDiscretisedDensity.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/IO/stir_ecat7.h"
#include "local/stir/decay_correct.h"
#include <iostream>
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"
#include <fstream>
#include "stir/IO/interfile.h"


#ifndef STIR_NO_NAMESPACES
using std::fstream;
using std::cerr;
#endif

START_NAMESPACE_STIR

// The assignement is necessary to prevent running the other than the copy-constructor. 
DynamicDiscretisedDensity::
DynamicDiscretisedDensity(const DynamicDiscretisedDensity& argument)
{
  (*this) = argument;
}

DynamicDiscretisedDensity&
DynamicDiscretisedDensity::
operator=(const DynamicDiscretisedDensity& argument)
{
  this->_time_frame_definitions = argument._time_frame_definitions;
  this->_densities.resize(argument._densities.size());
  for (unsigned int i=0; i<argument._densities.size(); ++i)
    this->_densities[i] = argument._densities[i]->clone();

  this->_scanner_sptr = argument._scanner_sptr;
  this->_calibration_factor = argument._calibration_factor;
  this->_isotope_halflife = argument._isotope_halflife;
  this->_is_decay_corrected = argument._is_decay_corrected; 
  return *this;
}

void 
DynamicDiscretisedDensity::
set_density_sptr(const shared_ptr<DiscretisedDensity<3,float> >& density_sptr, 
		 const unsigned int frame_num)
{  this->_densities[frame_num-1]=density_sptr; }  

const std::vector<shared_ptr<DiscretisedDensity<3,float> > > &
DynamicDiscretisedDensity::
get_densities() const 
{  return this->_densities ; }

const DiscretisedDensity<3,float> & 
DynamicDiscretisedDensity::
get_density(const unsigned int frame_num) const 
{  return *this->_densities[frame_num-1] ; }

DiscretisedDensity<3,float> & 
DynamicDiscretisedDensity::
get_density(const unsigned int frame_num)
{  return *this->_densities[frame_num-1] ; }

const float 
DynamicDiscretisedDensity::
get_isotope_halflife() const
{ return this->_isotope_halflife; }

const float  
DynamicDiscretisedDensity::
get_scanner_default_bin_size() const
{ return this->_scanner_sptr->get_default_bin_size(); }

const float  
DynamicDiscretisedDensity::
get_calibration_factor() const
{ return this->_calibration_factor; }

const TimeFrameDefinitions & 
DynamicDiscretisedDensity::
get_time_frame_definitions() const
{ return this->_time_frame_definitions; }

DynamicDiscretisedDensity*
DynamicDiscretisedDensity::
read_from_file(const string& filename) // The written image is read in respect to its center as origin!!!
{
  const int max_length=300;
  char signature[max_length];

  // read signature
  {
    fstream input(filename.c_str(), ios::in | ios::binary);
    if (!input)
      error("DynamicDiscretisedDensity::read_from_file: error opening file %s\n", filename.c_str());
    input.read(signature, max_length);
    signature[max_length-1]='\0';
  }

  DynamicDiscretisedDensity * dynamic_image_ptr = 0;
#ifdef HAVE_LLN_MATRIX
  dynamic_image_ptr =
      new DynamicDiscretisedDensity;

  if (strncmp(signature, "MATRIX", 6) == 0)
  {
#ifndef NDEBUG
    warning("DynamicDiscretisedDensity::read_from_file trying to read %s as ECAT7\n", filename.c_str());
#endif
    USING_NAMESPACE_ECAT
    USING_NAMESPACE_ECAT7

    if (is_ECAT7_image_file(filename))
    {
      Main_header mhead;
      if (read_ECAT7_main_header(mhead, filename) == Succeeded::no)
	{
	  warning("DynamicDiscretisedDensity::read_from_file cannot read %s as ECAT7\n", filename.c_str());
	  return 0;
	}
      dynamic_image_ptr->_scanner_sptr =
	find_scanner_from_ECAT_system_type(mhead.system_type);

      dynamic_image_ptr->_calibration_factor =
	mhead.calibration_factor;

      dynamic_image_ptr->_isotope_halflife =
	mhead.isotope_halflife;

      // TODO get this from the subheader fields or so
      // dynamic_image_ptr->_is_decay_corrected =
      //  shead.processing_code & DecayPrc
      dynamic_image_ptr->_is_decay_corrected = false;

      dynamic_image_ptr->_time_frame_definitions =
        TimeFrameDefinitions(filename);      

      dynamic_image_ptr->_densities.resize(dynamic_image_ptr->_time_frame_definitions.get_num_frames());

      const shared_ptr<VoxelsOnCartesianGrid<float>  > read_sptr = 
	    new VoxelsOnCartesianGrid<float> ; 

      for (unsigned int frame_num=1; frame_num <= (dynamic_image_ptr->_time_frame_definitions).get_num_frames(); ++ frame_num)
	{
	  dynamic_image_ptr->_densities[frame_num-1] =
	    ECAT7_to_VoxelsOnCartesianGrid(filename,
					   frame_num, 
	 /* gate_num, data_num, bed_num */ 1,0,0) ;
	}
      if (is_null_ptr(dynamic_image_ptr->_densities[0]))
	      error("DynamicDiscretisedDensity: None frame available\n");
    }
    else
    {
      if (is_ECAT7_file(filename))
	warning("DynamicDiscretisedDensity::read_from_file ECAT7 file %s should be an image\n", filename.c_str());
    }
  }
  else 
    error("DynamicDiscretisedDensity::read_from_file %s not corresponds to ECAT7 image\n");
#endif // end of HAVE_LLN_MATRIX
    // }    
  
  if (is_null_ptr(dynamic_image_ptr))   
  error("DynamicDiscretisedDensity::read_from_file %s pointer is NULL\n");
  return dynamic_image_ptr;
}

//Warning write_time_frame_definitions() is not yet implemented, so time information is missing.
/*	    sheader_ptr->frame_start_time=this->get_start_time(frame_num)*1000.;  //Start Time in Milliseconds
	    sheader_ptr->frame_duration=this->get_duration(frame_num)*1000.;	    //Duration in Milliseconds */
Succeeded 
DynamicDiscretisedDensity::
write_to_ecat7(const string& filename) const 
{
#ifndef HAVE_LLN_MATRIX
  return Succeeded::no;
#else

  Main_header mhead;
  ecat::ecat7::make_ECAT7_main_header(mhead, *_scanner_sptr, filename, get_density(1) );
  mhead.num_frames = (_time_frame_definitions).get_num_frames();
  mhead.acquisition_type =
    mhead.num_frames>1 ? DynamicEmission : StaticEmission;
  mhead.calibration_factor=_calibration_factor;
  mhead.isotope_halflife=_isotope_halflife;
  MatrixFile* mptr= matrix_create (filename.c_str(), MAT_CREATE, &mhead);
  if (mptr == 0)
    {
      warning("DynamicDiscretisedDensity::write_to_ecat7 cannot write output file %s\n", filename.c_str());
      return Succeeded::no;
    }
  for (  unsigned int frame_num = 1 ; frame_num<=(_time_frame_definitions).get_num_frames() ;  ++frame_num ) 
    {
      if (ecat::ecat7::DiscretisedDensity_to_ECAT7(mptr,
						   get_density(frame_num),
						   frame_num)
	  == Succeeded::no)
      {
        matrix_close(mptr);
        return Succeeded::no;
      }
    }
  matrix_close(mptr);
  return Succeeded::yes;
#endif // end of HAVE_LLN_MATRIX
}

 void DynamicDiscretisedDensity::
 calibrate_frames() const 
{
  for (  unsigned int frame_num = 1 ; frame_num<=(_time_frame_definitions).get_num_frames() ;  ++frame_num ) 
    {
      *(_densities[frame_num-1])*=_calibration_factor;
    }
}

void  DynamicDiscretisedDensity::
set_calibration_factor(const float calibration_factor) 
{ _calibration_factor=calibration_factor; }

void  DynamicDiscretisedDensity::
set_if_decay_corrected(const bool is_decay_corrected) 
{  this->_is_decay_corrected=is_decay_corrected; }

void  DynamicDiscretisedDensity::
set_isotope_halflife(const float isotope_halflife) 
{ _isotope_halflife=isotope_halflife; }

 void DynamicDiscretisedDensity::
 decay_correct_frames()  
{
  if (_is_decay_corrected==true)
    warning("DynamicDiscretisedDensity is already decay corrected");
  else
    {
      for (  unsigned int frame_num = 1 ; frame_num<=(_time_frame_definitions).get_num_frames() ;  ++frame_num ) 
	{ 
	  *(_densities[frame_num-1])*=decay_correct_factor(_isotope_halflife,_time_frame_definitions.get_start_time(frame_num),_time_frame_definitions.get_end_time(frame_num));	
	}
      _is_decay_corrected=true;
    }
}

END_NAMESPACE_STIR
