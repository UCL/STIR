//
// $Id$
//
/*
    Copyright (C) 2005 - 2011 Hammersmith Imanet Ltd
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
  \ingroup modelling
  \brief Implementations of inline functions of class stir::PlasmaData

  \author Charalampos Tsoumpas

  $Date$
  $Revision$
*/
#include "stir/decay_correction_factor.h"
#include "stir/numerics/integrate_discrete_function.h"

START_NAMESPACE_STIR

//! default constructor
PlasmaData::PlasmaData()
{ 
  this->set_if_decay_corrected(false);
} 

//! constructor giving a vector 
//ChT::ToDO: Better to use iterators
PlasmaData::PlasmaData(const std::vector<PlasmaSample> & plasma_blood_plot)
{
  this->_plasma_blood_plot=plasma_blood_plot;  
  this->set_if_decay_corrected(false); 
  this->_isotope_halflife=-1.;
}

//! default destructor
PlasmaData::~PlasmaData()
{ }

//! Implementation to read the input function from ONLY a 3-columns data file (Time-InputFunctionRadioactivity-WholeBloodRadioactivity).
void  PlasmaData::read_plasma_data(const std::string input_string) 
{ 
  std::ifstream data_stream(input_string.c_str()); 
  if(!data_stream)    
    error("cannot read plasma data from file.\n");    
  else
    data_stream >> _sample_size ;
  
  while(true)
    {
      float sample_time=0, blood_sample_radioactivity=0, plasma_sample_radioactivity=0;
      data_stream >> sample_time ;
      data_stream >> plasma_sample_radioactivity ;
      data_stream >> blood_sample_radioactivity ;
      if(!data_stream) 
        break;
      const PlasmaSample current_sample(sample_time,plasma_sample_radioactivity,blood_sample_radioactivity);
      (this->_plasma_blood_plot).push_back(current_sample);                          
      // Comment: The input function is generally not corrected for decay.
      this->set_if_decay_corrected(false);
    }
}     

// Implementation to set the input units not currently used.
/*
  void
       PlasmaData::set_input_units( SamplingTimeUnits input_sampling_time_units, 
                                    VolumeUnits input_volume_units, 
                                    RadioactivityUnits input_radioactivity_units )
{
  _input_sampling_time_units=input_sampling_time_units ;
  _input_volume_units=input_volume_units ;
  _input_radioactivity_units=input_radioactivity_units ;
} 
*/

//! Function to set the plasma_blood_plot 
void PlasmaData::set_plot(const std::vector<PlasmaSample> & plasma_blood_plot)
{  this->_plasma_blood_plot = plasma_blood_plot; }

//!Function to shift the time data
void PlasmaData::shift_time(const double time_shift)
{       
  _time_shift=time_shift;
  for(std::vector<PlasmaSample>::iterator cur_iter=this->_plasma_blood_plot.begin() ;
      cur_iter!=this->_plasma_blood_plot.end() ; ++cur_iter)
    cur_iter->set_time_in_s(cur_iter->get_time_in_s()+time_shift);                           
}

//!Function to get the time shift
double PlasmaData::get_time_shift()
{  return PlasmaData::_time_shift ; }

//!Function to get the isotope halflife
double
PlasmaData::get_isotope_halflife() const
{ return this->_isotope_halflife; }

//!Function to set the isotope halflife
void  PlasmaData::set_isotope_halflife(const double isotope_halflife) 
{ this->_isotope_halflife=isotope_halflife; }

void  PlasmaData::
set_time_frame_definitions(const TimeFrameDefinitions & plasma_fdef)
{  this->_plasma_fdef=plasma_fdef; }

TimeFrameDefinitions PlasmaData::
get_time_frame_definitions() const
{  return this->_plasma_fdef; }

void 
PlasmaData::
set_if_decay_corrected(const bool is_decay_corrected) 
{  this->_is_decay_corrected=is_decay_corrected; }

bool  
PlasmaData::
get_if_decay_corrected() const
{  return this->_is_decay_corrected; }

void 
PlasmaData::
decay_correct_PlasmaData()  
{
            
  if (this->_is_decay_corrected==true)
    warning("PlasmaData are already decay corrected");
  else
    {
      assert(this->_isotope_halflife>0);
      for(std::vector<PlasmaSample>::iterator cur_iter=this->_plasma_blood_plot.begin() ;
          cur_iter!=this->_plasma_blood_plot.end() ; ++cur_iter)
        {
          cur_iter->set_plasma_counts_in_kBq( static_cast<float>(cur_iter->get_plasma_counts_in_kBq()*decay_correction_factor(_isotope_halflife,cur_iter->get_time_in_s())));
          cur_iter->set_blood_counts_in_kBq( static_cast<float>(cur_iter->get_blood_counts_in_kBq()*decay_correction_factor(_isotope_halflife,cur_iter->get_time_in_s())));  
        }       
      PlasmaData::set_if_decay_corrected(true);
    }
}

//! Sorts the plasma_data into frames
PlasmaData 
PlasmaData::get_sample_data_in_frames(TimeFrameDefinitions time_frame_def)
{ 
  if (this->_is_decay_corrected==false)
    {
      this->decay_correct_PlasmaData();
      warning("Correcting for decay while sampling into frames.");
      this->set_if_decay_corrected(true);
    }
  std::vector<double> start_times_vector ;
  std::vector<double> durations_vector ;
  const unsigned int num_frames=time_frame_def.get_num_frames();
  std::vector<PlasmaSample> samples_in_frames_vector(num_frames);
  PlasmaData::const_iterator cur_iter;
  std::vector<PlasmaSample>::iterator frame_iter=samples_in_frames_vector.begin();

  // Estimate the plasma_frame_vector and the plasma_frame_sum_vector using the integrate_discrete_function() implementation
  for (unsigned int frame_num=1; frame_num<=num_frames && frame_iter!=samples_in_frames_vector.end() ; ++frame_num, ++frame_iter )
    {     
      std::vector<double> time_frame_vector ; 
      std::vector<double> plasma_frame_vector ;
      std::vector<double> blood_frame_vector ;
      const double frame_start_time=time_frame_def.get_start_time(frame_num);//t1
      const double frame_end_time=time_frame_def.get_end_time(frame_num);//t2

      for(cur_iter=(this->_plasma_blood_plot).begin() ; cur_iter!=(this->_plasma_blood_plot).end() ; ++cur_iter)
        {
          const double cur_time=(*cur_iter).get_time_in_s() ;
          if (cur_time<frame_start_time)
            continue;
          const double cur_plasma_cnt=(*cur_iter).get_plasma_counts_in_kBq();
          const double cur_blood_cnt=(*cur_iter).get_blood_counts_in_kBq();
          if (cur_time<frame_end_time) 
            {
              plasma_frame_vector.push_back(cur_plasma_cnt);
              blood_frame_vector.push_back(cur_blood_cnt);
              time_frame_vector.push_back(cur_time);        
            }
          else
            {
              if(plasma_frame_vector.size()<1) /* In case of no plasma data inside a frame, e.g. when there is large time_shift. */
                {
                  plasma_frame_vector.push_back(0.);
                  blood_frame_vector.push_back(0.);
                  time_frame_vector.push_back((frame_start_time+frame_end_time)*.5);
                }
              else
                {
                  plasma_frame_vector.push_back(cur_plasma_cnt);
                  blood_frame_vector.push_back(cur_blood_cnt);
                  time_frame_vector.push_back(cur_time);            
                }
              break;
            }     
        }
      if(time_frame_vector.size()!=1)
        {
          frame_iter->set_blood_counts_in_kBq(
			  static_cast<float>(integrate_discrete_function(time_frame_vector,blood_frame_vector)/
      		                     (time_frame_vector[time_frame_vector.size()-1]-time_frame_vector[0]))) ;
          frame_iter->set_plasma_counts_in_kBq(
             static_cast<float>(integrate_discrete_function(time_frame_vector,plasma_frame_vector)/
                                (time_frame_vector[time_frame_vector.size()-1]-time_frame_vector[0])));
          frame_iter->set_time_in_s(0.5*(time_frame_vector[time_frame_vector.size()-1]+time_frame_vector[0]));
          start_times_vector.push_back(time_frame_vector[0]); 
          durations_vector.push_back(time_frame_vector[time_frame_vector.size()-1]-time_frame_vector[0]) ;
        }
      else if(time_frame_vector.size()==1)
        {
          frame_iter->set_plasma_counts_in_kBq( static_cast<float>(plasma_frame_vector[0]));
          frame_iter->set_blood_counts_in_kBq( static_cast<float>(blood_frame_vector[0]));
          frame_iter->set_time_in_s(time_frame_vector[0]);
          start_times_vector.push_back(frame_start_time);
          durations_vector.push_back(frame_end_time-frame_start_time) ;
        }
    }
        PlasmaData plasma_data_in_frames(samples_in_frames_vector);
        TimeFrameDefinitions plasma_fdef(start_times_vector,durations_vector);
        plasma_data_in_frames.set_if_decay_corrected(this->_is_decay_corrected);
        plasma_data_in_frames.set_isotope_halflife(this->_isotope_halflife);
        plasma_data_in_frames.set_time_frame_definitions(plasma_fdef);
        return plasma_data_in_frames;
}

//PlasmaData begin() and end() of the PlasmaData ;
PlasmaData::const_iterator
PlasmaData::begin() const
{ return this->_plasma_blood_plot.begin() ; }

PlasmaData::const_iterator
PlasmaData::end() const
{ return this->_plasma_blood_plot.end() ; }

unsigned int
PlasmaData::size() const
{ return static_cast<unsigned>(this->_plasma_blood_plot.size()) ; }

/*
//PlasmaData begin() and end() of the PlasmaData ;
PlasmaData::iterator
PlasmaData::begin() 
{ return this->_plasma_blood_plot.begin() ; }

PlasmaData::iterator
PlasmaData::end() 
{ return this->_plasma_blood_plot.end() ; }
*/

END_NAMESPACE_STIR
