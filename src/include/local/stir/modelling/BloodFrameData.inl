//
// $Id$
//
/*
  Copyright (C) 2005 - $Date$, Hammersmith Imanet Ltd
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

  \file
  \ingroup modelling

  \brief Implementations of inline functions of class stir::BloodFrameData

  \author Charalampos Tsoumpas

  $Date$
  $Revision$
*/

START_NAMESPACE_STIR

//! default constructor
BloodFrameData::BloodFrameData()
{ } 

//! constructor giving a vector //ChT::ToDO: Better to use iterators
BloodFrameData::BloodFrameData(const std::vector<BloodFrame> & blood_plot)
{this->_blood_plot=blood_plot;}

//! default destructor
BloodFrameData::~BloodFrameData()
{ }

//! Implementation to read the input function from ONLY a 2-columns frame data (FrameNumber-InputFunctionRadioactivity).
void  BloodFrameData::read_blood_frame_data(const std::string input_string) 
{
  std::ifstream data_stream(input_string.c_str()); 
  if(!data_stream)    
    error("cannot read blood frame data from file.\n");    
  else
    {
      data_stream >> _num_frames ;  
      while(true)
	{
	  unsigned int frame_num=0;
	  float blood_sample_radioactivity=0.F;
	  data_stream >> frame_num ;
	  data_stream >> blood_sample_radioactivity ;
	  if(!data_stream) 
	    break;
	  const BloodFrame current_sample(frame_num,blood_sample_radioactivity);
	  (this->_blood_plot).push_back(current_sample);		     	     
	}
    }  
}

//! Implementation to set the input units not currently used.
void
BloodFrameData::set_input_units( SamplingTimeUnits input_sampling_time_units, 
				 VolumeUnits input_volume_units, 
				 RadioactivityUnits input_radioactivity_units )
{
  _input_sampling_time_units=input_sampling_time_units ;
  _input_volume_units=input_volume_units ;
  _input_radioactivity_units=input_radioactivity_units ;
} 

//!Function to shift the time data
void BloodFrameData::shift_time(const float time_shift)
{	
  _time_shift=time_shift;
  for(std::vector<BloodFrame>::iterator cur_iter=this->_blood_plot.begin() ;
      cur_iter!=this->_blood_plot.end() ; ++cur_iter)
    {
      cur_iter->set_frame_start_time_in_s(cur_iter->get_frame_start_time_in_s()+time_shift);
      cur_iter->set_frame_end_time_in_s(cur_iter->get_frame_end_time_in_s()+time_shift);
    }
}
//!Function to get the time data
float BloodFrameData::get_time_shift()
{  return BloodFrameData::_time_shift ; }

void  BloodFrameData::set_isotope_halflife(const float isotope_halflife) 
{ _isotope_halflife=isotope_halflife; }

void  BloodFrameData::
set_if_decay_corrected(const bool is_decay_corrected) 
{  this->_is_decay_corrected=is_decay_corrected; }

void BloodFrameData::
decay_correct_BloodFrameData()  
{	    
  if (BloodFrameData::_is_decay_corrected==true)
    warning("BloodFrameData are already decay corrected");
  else
    {
      for(std::vector<BloodFrame>::iterator cur_iter=this->_blood_plot.begin() ;
	  cur_iter!=this->_blood_plot.end() ; ++cur_iter)
	 cur_iter->set_blood_counts_in_kBq(cur_iter->get_blood_counts_in_kBq()
					  *decay_correct_factor(_isotope_halflife,cur_iter->get_frame_start_time_in_s(),cur_iter->get_frame_end_time_in_s()));
      BloodFrameData::set_if_decay_corrected(true);
    }
}

//BloodFrameData begin() and end() of the BloodFrameData ;
BloodFrameData::const_iterator
BloodFrameData::begin() const
{ return this->_blood_plot.begin() ; }

BloodFrameData::const_iterator
BloodFrameData::end() const
{ return this->_blood_plot.end() ; }

/*
//BloodFrameData begin() and end() of the BloodFrameData ;
BloodFrameData::iterator
BloodFrameData::begin() 
{ return this->_blood_plot.begin() ; } BloodFrameData::iterator BloodFrameData::end() { return this->_blood_plot.end() ; } */


END_NAMESPACE_STIR
