//
// $Id$
//
/*
    Copyright (C) 2004 - $Date$, Hammersmith Imanet Ltd
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

  \brief Implementations of inline functions of class stir::PlasmaData

  \author Charalampos Tsoumpas

  $Date$
  $Revision$
*/

START_NAMESPACE_STIR

  //! default constructor
PlasmaSample::PlasmaSample()
{ }
  //! constructor
PlasmaSample::PlasmaSample(const float sample_time, const float plasma_sample_counts, const float blood_sample_counts)
{
  PlasmaSample::set_time_in_s( sample_time );
  PlasmaSample::set_blood_counts_in_kBq( blood_sample_counts );  
  PlasmaSample::set_plasma_counts_in_kBq( plasma_sample_counts );  
}
  //! default destructor
PlasmaSample::~PlasmaSample()
{ }
  
  //! set the time of the sample
void PlasmaSample::set_time_in_s( const float time )
{ PlasmaSample::_time=time ; }
  //! get the time of the sample
float PlasmaSample::get_time_in_s() const
{  return PlasmaSample::_time ; }
  //! set the blood counts of the sample 
void PlasmaSample::set_blood_counts_in_kBq( const float blood_counts )
{ PlasmaSample::_blood_counts=blood_counts ; }
  //! get the blood counts of the sample 
float PlasmaSample::get_blood_counts_in_kBq() const
{  return PlasmaSample::_blood_counts ; }
  //! get the plasma counts of the sample 
void PlasmaSample::set_plasma_counts_in_kBq( const float plasma_counts )
{ PlasmaSample::_plasma_counts=plasma_counts ; }
  //! get the plasma counts of the sample 
float PlasmaSample::get_plasma_counts_in_kBq() const
{  return PlasmaSample::_plasma_counts ; }

  //! default constructor
PlasmaData::PlasmaData()
{ } 
  //! default destructor
PlasmaData::~PlasmaData()
{ }

//! Implementation to read the input function from ONLY a 3-columns data file (Time-InputFunctionRadioactivity-WholeBloodRadioactivity).
void  PlasmaData::read_plasma_data(const std::string input_string) 
{ 
  std::ifstream data_stream(input_string.c_str()); 
  if(!data_stream)    
    std::cerr << "Cannot open " << input_string << std::endl ;
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
	(this->_plasma_plot).push_back(current_sample);		     	     
      }
}     
  //! Implementation to set the input units not currently used.
void
 PlasmaData::set_input_units( SamplingTimeUnits input_sampling_time_units, 
		        VolumeUnits input_volume_units, 
		        RadioactivityUnits input_radioactivity_units )
{
  _input_sampling_time_units=input_sampling_time_units ;
  _input_volume_units=input_volume_units ;
  _input_radioactivity_units=input_radioactivity_units ;
} 

  //!Function to shift the time data
  void PlasmaData::shift_time(float time_shift)
{	
	_time_shift=time_shift;
	for(std::vector<PlasmaSample>::iterator cur_iter=this->_plasma_plot.begin() ;
	    cur_iter!=this->_plasma_plot.end() ; ++cur_iter)
	  cur_iter->set_time_in_s(cur_iter->get_time_in_s()+time_shift);		     	     
}
  //!Function to get the time data
  float PlasmaData::get_time_shift()
{  return PlasmaData::_time_shift ; }


//PlasmaData begin() and end() of the PlasmaData ;
PlasmaData::const_iterator
PlasmaData::begin() const
{ return this->_plasma_plot.begin() ; }

PlasmaData::const_iterator
PlasmaData::end() const
{ return this->_plasma_plot.end() ; }


END_NAMESPACE_STIR
