//
//
/*
    Copyright (C) 2005 - 2011, Hammersmith Imanet Ltd
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
 
*/

#ifndef __stir_modelling_PlasmaData_H__
#define __stir_modelling_PlasmaData_H__

#include "stir/modelling/PlasmaSample.h"
#include "stir/TimeFrameDefinitions.h"
#include <vector>

START_NAMESPACE_STIR

//! A class for storing plasma samples of a single study.
/*! \ingroup modelling
 */
class PlasmaData 
{
 typedef std::vector<PlasmaSample> plot_type;
 
 public: 
  //! constructor giving a vector \todo Better to use iterators
  inline PlasmaData(const std::vector<PlasmaSample> & plasma_blood_plot);
  inline PlasmaData();   //!< default constructor
  inline ~PlasmaData();   //!< default constructor

 typedef plot_type::const_iterator const_iterator;
  //!\todo Implementation to set the input units.

 /*
  enum VolumeUnits 
    { ml , litre };  
  enum SamplingTimeUnits
    { seconds , minutes };
  enum RadioactivityUnits
    { counts_per_sec , counts_per_min , kBq };
 */

  // Implementation to set the input units not currently used. Always, it assumed to use kBq, seconds, ml.
  /*  inline void set_input_units(const SamplingTimeUnits input_sampling_time_units, 
                              const VolumeUnits input_volume_units, 
                              const RadioactivityUnits input_radioactivity_units ) ;
  */


 /*! Implementation to read the input function from ONLY a 3-columns data file (Time-InputFunctionRadioactivity-TotalBloodRadioactivity).
   \warning Assumes that the input function is not corrected for decay.
 */
  inline void read_plasma_data(const std::string input_string) ;

  inline void set_plot(const std::vector<PlasmaSample> & plasma_blood_plot);

  /*! Sorts the plasma_data into frames
    \warning It corrects for decay if the data are not decay corrected. 
    \return PlasmaData are in start-end frames time mode. 
  */
  inline PlasmaData get_sample_data_in_frames(TimeFrameDefinitions time_frame_def);

  /*!Function to shift the time data
    This is useful if the start time of the scan and the start time of the plasma are not precisely correct. 
    This can be measured by the plasma peak and the very first frames of the dynamic images.   
    \note This cannot be estimated in the current implementation of the direct reconstructions. Thus, it is given externally. */

   //! \name Functions to get parameters @{
  inline double get_time_shift();
  inline bool get_if_decay_corrected() const ;
  inline double get_isotope_halflife() const;  
  inline TimeFrameDefinitions get_time_frame_definitions() const;  //!@}
  //! \name Functions to set parameters 
  //!@{
  inline void set_time_frame_definitions(const TimeFrameDefinitions & plasma_fdef);   //!<\note The set_time_frame_definitions() is prefered than giving directly the Scan TimeFrameDefinitions since the sample may not be measured for all the frames \n For example at the beginning or at the end of the scan.
  inline void set_if_decay_corrected(const bool is_decay_corrected);
  inline void set_isotope_halflife(const double isotope_halflife);  
  inline void shift_time(const double time_shift);
  //!@}

  //!Function to decay correct the data
  inline void decay_correct_PlasmaData();

  //!  begin() and end() iterators for the plasma curve and the size() function 
  //@{
  inline const_iterator begin() const ;
  inline const_iterator end() const ;
  inline unsigned int size() const ;
  //!@}

  //!\todo non const_iterator should be defined if the plasma data needs to be changed 
  //inline iterator begin() ;
  //inline iterator end()  ;

 private:
  //  PlasmaSample _plasma_type ;
  //  VolumeUnits _input_volume_units ; 
  //  SamplingTimeUnits _input_sampling_time_units ;
  //  RadioactivityUnits _input_radioactivity_units ;
  bool _is_decay_corrected ;
  std::vector<PlasmaSample> _plasma_blood_plot ;
  TimeFrameDefinitions _plasma_fdef;
  double _isotope_halflife;
  int _sample_size;
  double _time_shift ;
};

END_NAMESPACE_STIR

#include "stir/modelling/PlasmaData.inl"

#endif //__stir_modelling_PlasmaData_H__
