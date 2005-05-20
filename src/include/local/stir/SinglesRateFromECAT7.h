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
  \ingroup local_buildblock

  \brief Declaration of class SinglesRatesFromECAT7
  \todo file-name is incorrect (misses an s)
  \author  Sanida Mustafovic and Kris Thielemans
  $Date$
  $Revision$ 
*/

#ifndef __stir_SinglesRatesFromECAT7_H__
#define __stir_SinglesRatesFromECAT7_H__

#include "local/stir/SinglesRates.h"
#include "stir/Array.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/IO/stir_ecat7.h"
#include "stir/TimeFrameDefinitions.h"

START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7

class SinglesRatesFromECAT7 : 
public RegisteredParsingObject<SinglesRatesFromECAT7, SinglesRates>

{ 
public:

    //! Name which will be used when parsing a SinglesRatesFromECAT7 object 
    static const char * const registered_name; 
    
    //! Default constructor 
    SinglesRatesFromECAT7 ();

    //! Given the detection position get the singles rate   
    virtual float get_singles_rate(const DetectionPosition<>& det_pos, 
                                   const double start_time,
                                   const double end_time) const;


    
    //! Generate a FramesSinglesRate - containing the average rates
    //  for a frame begining at start_time and ending at end_time.
    FrameSinglesRates get_rates_for_frame(double start_time,
                                          double end_time) const;
    
    
    int get_frame_number (const double start_time, const double end_time) const;

    /*
    // Get a starting and ending frame number for the specified time interval.
    // The starting and ending frames will stretch from the first frame in
    // which start_time is included to the first frame in which end_time is
    // included. Therefore the specified time interval will not usually cover
    // the entire time of all frames between start_frame and end_frame.
    // If the specified time interval falls outside the set of frames then
    // both start_time and end_time are set to zero.
    void get_frame_interval(double start_time, double end_time,
                            int& start_frame, int& end_frame) const;
    */
    
    //!  The function that reads singles from ECAT7 file
    int read_singles_from_file(const string& ECAT7_filename,
                               const std::ios::openmode open_mode = std::ios::in);
    
   
private:

  Array<2,float> singles;

  string ECAT7_filename;
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();
  
  float  SinglesRatesFromECAT7::get_singles_rate(int singles_bin_index,
                                                 double start_time,
                                                 double end_time) const;
  
 
  TimeFrameDefinitions time_frame_defs;


  
};

END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
END_NAMESPACE_STIR


#endif
