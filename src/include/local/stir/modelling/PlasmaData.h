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
*/
/*!
  \file
  \ingroup modelling

  \brief Declaration of class stir::PlasmaData

  \author Charalampos Tsoumpas
 
  $Date$
  $Revision$
*/

#ifndef __stir_modelling_PlasmaData_H__
#define __stir_modelling_PlasmaData_H__

#include "local/stir/modelling/PlasmaSample.h"
#include <vector>

START_NAMESPACE_STIR


/*!   
   \ingroup modelling
   \brief
 A class for storing plasma samples of a single study.
*/

class PlasmaData
{
 typedef std::vector<PlasmaSample> plot_type;
 
 public: 
 typedef plot_type::const_iterator const_iterator;
  enum VolumeUnits 
    { ml , litre };  
  enum SamplingTimeUnits
    { seconds , minutes };
  enum RadioactivityUnits
    { counts_per_sec , counts_per_min , kBq };

  //! Implementation to read the input function from ONLY a 3-columns data file (Time-InputFunctionRadioactivity-TotalBloodRadioactivity).
  inline void read_plasma_data(const std::string input_string) ;
  //! Implementation to set the input units not currently used. Always, it assumed to use kBq, seconds, ml.
  inline void set_input_units(const SamplingTimeUnits input_sampling_time_units, 
			      const VolumeUnits input_volume_units, 
			      const RadioactivityUnits input_radioactivity_units ) ;

  //!Function to shift the time data
  inline void shift_time(float time_shift);

  //!Function to get the time data
  inline float get_time_shift();

  //!Function to set the isotope halflife
  inline void set_isotope_halflife(const float isotope_halflife);

  //!Function to set _is_decay_corrected boolean true ar false
  inline void set_if_decay_corrected(const bool is_decay_corrected);

  //!Function to decay correct the data
  inline void decay_correct_PlasmaData();

  //! default constructor
  inline PlasmaData();

  //! constructor giving a vector //ChT::ToDO: Better to use iterators
  inline PlasmaData(std::vector<PlasmaSample> plasma_blood_plot);

  //! default constructor
  inline ~PlasmaData();

  //!  void begin() and end() iterators for the plasma curve ;
inline const_iterator begin() const ;
inline const_iterator end() const ;
  // non const_iterator should be defined if the plasma data needs to be changed 
//inline iterator begin() ;
//inline iterator end()  ;
  
 private:
  PlasmaSample _plasma_type ;
  VolumeUnits _input_volume_units ; 
  SamplingTimeUnits _input_sampling_time_units ;
  RadioactivityUnits _input_radioactivity_units ;
  bool _is_decay_corrected ;
  float _isotope_halflife;
  std::vector<PlasmaSample> _plasma_blood_plot ;
  int _sample_size;
  float _time_shift ;
};


END_NAMESPACE_STIR


#include "local/stir/modelling/PlasmaData.inl"

#endif //__stir_modelling_PlasmaData_H__
