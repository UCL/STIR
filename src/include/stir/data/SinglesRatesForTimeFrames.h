//
//
/*
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
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

  \brief Declaration of class stir::SinglesRatesForTimeFrames
  \author Kris Thielemans
  \author Timothy Borgeaud
*/

#ifndef __stir_data_SinglesRatesForTimeFrames_H__
#define __stir_data_SinglesRatesForTimeFrames_H__

#include "stir/data/SinglesRates.h"
#include "stir/Array.h"
#include "stir/TimeFrameDefinitions.h"

START_NAMESPACE_STIR

/*!
  \ingroup singles_buildblock
  \brief A class for singles rates that are recorded in time frames.
*/
class SinglesRatesForTimeFrames 
: public SinglesRates
{ 
public:
  
    //! Default constructor 
    SinglesRatesForTimeFrames ();

    //SinglesRatesForTimeFrames(const TimeFrameDefinitions& time_frame_definitions,
    //			      const shared_ptr<Scanner>& scanner_sptr);


    //! get the singles rate for a particular singles unit and frame number.   
    //
    // The singles rate returned is the rate for a whole singles unit.
    //
    float get_singles_rate(int singles_bin_index, unsigned int frame_number) const;
    
    
    //! get the singles rate for a particular singles unit and a frame with 
    // the specified start and end times.   
    //
    // The singles rate returned is the rate for a whole singles unit.
    //
    /*! \warning Currently returns -1 if the \a start_time, \a end_time
        does not corresponds to a time frame.
    */
    float get_singles_rate(const int singles_bin_index, 
                           const double start_time, const double end_time) const;
    
    //! Generate a FramesSinglesRate - containing the average rates
    //  for a frame begining at start_time and ending at end_time.
    FrameSinglesRates get_rates_for_frame(double start_time,
                                          double end_time) const;
    
    //! Set a singles rate by singles bin index and time frame number.
    /*! \warning No error checking is doing on validity of the indices.
     */
    void set_singles_rate(const int singles_bin_index, 
			  const unsigned time_frame_num, 
			  const float new_rate);
    
    //! Get the number of frames for which singles rates are recorded.
    unsigned int get_num_frames() const;

    //! Get the time frame definitions
    const TimeFrameDefinitions& 
      get_time_frame_definitions() const;

protected:

  Array<2,float> _singles; 
  TimeFrameDefinitions _time_frame_defs;


  
};

END_NAMESPACE_STIR


#endif
