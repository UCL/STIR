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
*/
/*!
  \file
  \ingroup modelling

  \brief Declaration of class stir::PlasmaData

  \author Charalampos Tsoumpas
 
  $Date$
  $Revision$
*/

#ifndef __stir_PlasmaSampling_H__
#define __stir_PlasmaSampling_H__


#include "stir/common.h"
#include <vector>
#include <iostream> 
#include <cstring>
#include <iomanip> 
#include <fstream>



START_NAMESPACE_STIR


class PlasmaSample
{ 
public:
   //! default constructor
  inline PlasmaSample();

  //!  A constructor : constructs a PlasmaSample object  
  inline PlasmaSample( const float sample_time, const float plasma_sample_counts, const float blood_sample_counts);

  //! default constructor
  inline ~PlasmaSample();
   
 //! set the time of the sample
  inline void set_time_in_s( const float time );
 //! get the time of the sample
  inline float get_time_in_s() const; 
 //! set the blood counts of the sample
  inline void set_blood_counts_in_kBq( const float blood_counts );
 //! get the blood counts of the sample
  inline float get_blood_counts_in_kBq() const; 
 //! set the plasma counts of the sample
  inline void set_plasma_counts_in_kBq( const float plasma_counts );
 //! get the plasma counts of the sample
  inline float get_plasma_counts_in_kBq() const; 
  
private : 
  float _blood_counts;
  float _plasma_counts;
  float _time;
};

/*!   
   \ingroup modelling
   \brief
 A class for storing plasma samples of a single study.
*/

class PlasmaData:PlasmaSample
{
 typedef std::vector<PlasmaSample> plot_type;
 
 public: 
typedef plot_type::const_iterator const_iterator;
  enum PlasmaType 
    { arterial , arterialized , venous };
  enum VolumeUnits 
    { ml , litre };
  enum SamplingType
    { non_regular , regular };
  enum SamplingTimeUnits
    { seconds , minutes };
  enum RadioactivityUnits
    { counts_per_sec , counts_per_min , kBq };

  //! Implementation to read the input function from ONLY a 3-columns data file (Time-InputFunctionRadioactivity-TotalBloodRadioactivity).
  inline void read_plasma_data(const std::string input_string) ;
  //! Implementation to set the input units not currently used.
  inline void set_input_units(const SamplingTimeUnits input_sampling_time_units, 
			      const VolumeUnits input_volume_units, 
			      const RadioactivityUnits input_radioactivity_units ) ;
  //!Function to shift the time data
  inline void shift_time(float time_shift);

  //!Function to get the time data
  inline float get_time_shift();

  //! default constructor
  inline PlasmaData();

  //! default constructor
  inline ~PlasmaData();

  //!  void begin() and end() iterators for the plasma curve ;
inline const_iterator begin() const ;
inline const_iterator end() const ;
  // non const_iterator should be defined if the plasma data needs to be changed 

  
 private:
  PlasmaType _plasma_type ;
  VolumeUnits _input_volume_units ; 
  SamplingType _sampling_type ;
  SamplingTimeUnits _input_sampling_time_units ;
  RadioactivityUnits _input_radioactivity_units ;
  std::vector<PlasmaSample> _plasma_plot ;
  int _sample_size;
  float _time_shift ;
};


END_NAMESPACE_STIR


#include "local/stir/modelling/PlasmaData.inl"

#endif //__PlasmaSampling_H__
