//
// $Id$
//
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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
  \ingroup singles_buildblock

  \brief Declaration of class stir::ecat::ecat7::SinglesRatesFromECAT7
  \author  Sanida Mustafovic and Kris Thielemans
  $Date$
  $Revision$ 
*/

#ifndef __stir_data_SinglesRatesFromECAT7_H__
#define __stir_data_SinglesRatesFromECAT7_H__

#include "stir/data/SinglesRates.h"
#include "stir/Array.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/IO/stir_ecat7.h"
#include "stir/TimeFrameDefinitions.h"

START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7

/*!
  \ingroup singles_buildblock
  \brief A class that extracts singles info from an ECAT7 sinogram file.
*/
class SinglesRatesFromECAT7 : 
public RegisteredParsingObject<SinglesRatesFromECAT7, SinglesRates>

{ 
public:

    //! Name which will be used when parsing a SinglesRatesFromECAT7 object 
    static const char * const registered_name; 
    
    //! Default constructor 
    SinglesRatesFromECAT7 ();


    //! get the singles rate for a particular singles unit and frame number.   
    //
    // The singles rate returned is the rate for a whole singles unit.
    //
    float get_singles_rate(int singles_bin_index, int frame_number) const;
    
    
    //! get the singles rate for a particular singles unit and a frame with 
    // the specified start and end times.   
    //
    // The singles rate returned is the rate for a whole singles unit.
    //
    float get_singles_rate(int singles_bin_index, 
                           double start_time, double end_time) const;
    
    //! Given the detection position and frame start and end times, get the 
    // singles rate.   
    //
    // The singles rate returned is the rate for a whole singles unit.
    //
    virtual float get_singles_rate(const DetectionPosition<>& det_pos, 
                                   const double start_time,
                                   const double end_time) const;


    
    //! Generate a FramesSinglesRate - containing the average rates
    //  for a frame begining at start_time and ending at end_time.
    FrameSinglesRates get_rates_for_frame(double start_time,
                                          double end_time) const;
    
    //! Get the frame number associated with a frame starting and start_time and
    // ending at end_time.
    int get_frame_number(const double start_time, const double end_time) const;

    //! Get the number of frames for which singles rates are recorded.
    int get_num_frames() const;

    //! Get the frame start time for a particular frame.
    double get_frame_start(unsigned int frame_number) const;

    //! Get the frame end time for a particular frame.
    double get_frame_end(unsigned int frame_number) const;

    
    //!  The function that reads singles from ECAT7 file
    int read_singles_from_file(const string& ECAT7_filename,
                               const std::ios::openmode open_mode = std::ios::in);
    
   
private:

  Array<2,float> singles;

  string ECAT7_filename;
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();
  
 
 
  TimeFrameDefinitions time_frame_defs;


  
};

END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
END_NAMESPACE_STIR


#endif
