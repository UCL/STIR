//
//
/*
    Copyright (C) 2005 - 2007, Hammersmith Imanet Ltd
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

  \brief Declaration of class stir::BloodFrameData

  \author Charalampos Tsoumpas
 
*/

#ifndef __stir_modelling_BloodFrameData_H__
#define __stir_modelling_BloodFrameData_H__

#include "local/stir/modelling/BloodFrame.h"
#include "local/stir/decay_correct.h"
#include <vector>

START_NAMESPACE_STIR


/*!   
   \ingroup modelling
   \brief
 A class for storing plasma samples of a single study.
*/
class BloodFrameData
{
 typedef std::vector<BloodFrame> plot_type;
 
 public: 

 typedef plot_type::const_iterator const_iterator;
 /*  enum VolumeUnits 
    { ml , litre };
  enum SamplingTimeUnits
    { seconds , minutes };
  enum RadioactivityUnits
    { counts_per_sec , counts_per_min , kBq };

  inline void set_input_units(const SamplingTimeUnits input_sampling_time_units, 
			      const VolumeUnits input_volume_units, 
			      const RadioactivityUnits input_radioactivity_units ) ;

 */
  //! Implementation to read the input function from ONLY a 2-columns frame data (FrameNumber-InputFunctionRadioactivity).
  inline void read_blood_frame_data(const std::string input_string) ;
  inline void set_plot(const std::vector<BloodFrame> & blood_plot) ;
  //! Implementation to set the input units not currently used. Always, it assumed to use kBq, seconds, ml.

  //!Function to shift the time data
  inline void shift_time(const float time_shift);

  //!Function to get the time data
  inline float get_time_shift();

  //!Function to set the isotope halflife
  inline void set_isotope_halflife(const float isotope_halflife);

  //!Function to set _is_decay_corrected boolean true ar false
  inline void set_if_decay_corrected(const bool is_decay_corrected);

  //!Function to set _is_decay_corrected boolean true ar false
  inline bool get_if_decay_corrected();

  //!Function to decay correct the data
  inline void decay_correct_BloodFrameData();

  //! default constructor
  inline BloodFrameData();

  //! constructor giving a vector //ChT::ToDO: Better to use iterators
  inline BloodFrameData(const std::vector<BloodFrame> & blood_plot);

  //! default constructor
  inline ~BloodFrameData();

  //!  void begin() and end() iterators for the plasma curve ;
inline const_iterator begin() const ;
inline const_iterator end() const ;
 inline unsigned int size() const ;
  // non const_iterator should be defined if the plasma data needs to be changed 
//inline iterator begin() ;
//inline iterator end()  ;
  
 private:
 /*  VolumeUnits _input_volume_units ; 
  SamplingTimeUnits _input_sampling_time_units ;
  RadioactivityUnits _input_radioactivity_units ;*/
  bool _is_decay_corrected ;
  float _isotope_halflife;
  std::vector<BloodFrame> _blood_plot ;
  int _num_frames;
  float _time_shift ;
};


END_NAMESPACE_STIR


#include "local/stir/modelling/BloodFrameData.inl"

#endif //__stir_modelling_BloodFrameData_H__
