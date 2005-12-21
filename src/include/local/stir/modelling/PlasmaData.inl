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

  \brief Implementations of inline functions of class stir::PlasmaData

  \author Charalampos Tsoumpas

  $Date$
  $Revision$
*/

START_NAMESPACE_STIR

  //! default constructor
PlasmaData::PlasmaData()
{ } 

  //! constructor giving a vector //ChT::ToDO: Better to use iterators
PlasmaData::PlasmaData(const std::vector<PlasmaSample> & plasma_blood_plot)
{this->_plasma_blood_plot=plasma_blood_plot;}

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
  void PlasmaData::shift_time(const float time_shift)
{	
	_time_shift=time_shift;
	for(std::vector<PlasmaSample>::iterator cur_iter=this->_plasma_blood_plot.begin() ;
	    cur_iter!=this->_plasma_blood_plot.end() ; ++cur_iter)
	  cur_iter->set_time_in_s(cur_iter->get_time_in_s()+time_shift);		     	     
}
  //!Function to get the time data
  float PlasmaData::get_time_shift()
{  return PlasmaData::_time_shift ; }

void  PlasmaData::set_isotope_halflife(const float isotope_halflife) 
{ _isotope_halflife=isotope_halflife; }

void  PlasmaData::
set_if_decay_corrected(const bool is_decay_corrected) 
{  this->_is_decay_corrected=is_decay_corrected; }

 void PlasmaData::
 decay_correct_PlasmaData()  
{
	    
  if (PlasmaData::_is_decay_corrected==true)
    warning("PlasmaData are already decay corrected");
  else
    {
	for(std::vector<PlasmaSample>::iterator cur_iter=this->_plasma_blood_plot.begin() ;
	    cur_iter!=this->_plasma_blood_plot.end() ; ++cur_iter)
	{
		cur_iter->set_plasma_counts_in_kBq(cur_iter->get_plasma_counts_in_kBq()*decay_correct_factor(_isotope_halflife,cur_iter->get_time_in_s()));
		cur_iter->set_blood_counts_in_kBq(cur_iter->get_blood_counts_in_kBq()*decay_correct_factor(_isotope_halflife,cur_iter->get_time_in_s()));	
	}	
	 PlasmaData::set_if_decay_corrected(true);
	}
}

//PlasmaData begin() and end() of the PlasmaData ;
PlasmaData::const_iterator
PlasmaData::begin() const
{ return this->_plasma_blood_plot.begin() ; }

PlasmaData::const_iterator
PlasmaData::end() const
{ return this->_plasma_blood_plot.end() ; }

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
