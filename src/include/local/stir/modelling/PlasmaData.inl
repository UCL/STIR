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

  \brief Implementations of inline functions of class stir::PlasmaSampling

  \author Charalampos Tsoumpas

  $Date$
  $Revision$
*/

START_NAMESPACE_STIR

  //! default constructor
PlasmaSample::PlasmaSample()
{ }
  //! constructor
PlasmaSample::PlasmaSample(const float sample_time, const float sample_counts)
{
  PlasmaSample::set_time_in_s( sample_time );
  PlasmaSample::set_counts_in_kBq( sample_counts );
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
  //! set the counts of the sample 
void PlasmaSample::set_counts_in_kBq( const float counts )
{ PlasmaSample::_counts=counts ; }
  //! get the counts of the sample 
float PlasmaSample::get_counts_in_kBq() const
{  return PlasmaSample::_counts ; }

  //! default constructor
PlasmaData::PlasmaData()
{ } 
  //! default destructor
PlasmaData::~PlasmaData()
{ }

void  PlasmaData::read_plasma_data(const std::string input_string) 
{ 
  std::ifstream data_stream(input_string.c_str()); 
  if(!data_stream)    
    std::cerr << "Cannot open " << input_string << std::endl ;
  else
    while(true)
      {
	float sample_time=0, sample_radioactivity=0;
	data_stream >> sample_time ;
	data_stream >> sample_radioactivity ;
	if(!data_stream) 
	  break;
	const PlasmaSample current_sample(sample_time,sample_radioactivity);
	(this->_plasma_plot).push_back(current_sample);		     	     
      }
}     
void
 PlasmaData::set_input_units( SamplingTimeUnits input_sampling_time_units, 
		        PlasmaVolumeUnits input_plasma_volume_units, 
		        RadioactivityUnits input_radioactivity_units )
{
  _input_sampling_time_units=input_sampling_time_units ;
  _input_plasma_volume_units=input_plasma_volume_units ;
  _input_radioactivity_units=input_radioactivity_units ;
} 

//PlasmaData begin() and end() of the PlasmaData ;
PlasmaData::const_iterator
PlasmaData::begin() const
{ return this->_plasma_plot.begin() ; }

PlasmaData::const_iterator
PlasmaData::end() const
{ return this->_plasma_plot.end() ; }


END_NAMESPACE_STIR
