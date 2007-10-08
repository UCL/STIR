//
// $Id$
//
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
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
  \brief Implementation of stir::SinglesRatesForTimeFrames

  \author Kris Thielemans
  \author Timothy Borgeaud
  $Date$
  $Revision$
*/

#include "stir/data/SinglesRatesForTimeFrames.h"
//#include "stir/IndexRange2D.h"

START_NAMESPACE_STIR

SinglesRatesForTimeFrames::
SinglesRatesForTimeFrames()
{}


float 
SinglesRatesForTimeFrames::
get_singles_rate(const int singles_bin_index, 
		 const unsigned int time_frame_num) const
{ 
  return(this->_singles[time_frame_num][singles_bin_index]);
}

void 
SinglesRatesForTimeFrames::
set_singles_rate(const int singles_bin_index, 
		 const unsigned time_frame_num, 
		 const float new_rate)

{
  this->_singles[time_frame_num][singles_bin_index] = new_rate;
}


float
SinglesRatesForTimeFrames::
get_singles_rate(const int singles_bin_index,
		 const double start_time,
		 const double end_time) const
{
  const unsigned frame_number = 
    this->_time_frame_defs.get_time_frame_num(start_time, end_time);
  if (frame_number == 0)
    return -1.F;
  else
  return(get_singles_rate(singles_bin_index, frame_number));
}

unsigned int 
SinglesRatesForTimeFrames::get_num_frames() const {
  return(this->_time_frame_defs.get_num_frames());
}

const TimeFrameDefinitions&
SinglesRatesForTimeFrames:: 
get_time_frame_definitions() const
{
  return this->_time_frame_defs;
}


#if 0
double 
SinglesRatesForTimeFrames::
get_frame_start(unsigned int frame_number) const {
  if ( frame_number < 1 || frame_number > this->_time_frame_defs.get_num_frames() ) {
    return(0.0);
  } else {
    return(this->_time_frame_defs.get_start_time(frame_number));
  }
}
 


double 
SinglesRatesForTimeFrames::
get_frame_end(unsigned int frame_number) const {
  if ( frame_number < 1 || frame_number > this->_time_frame_defs.get_num_frames() ) {
    return(0.0);
  } else {
    return(this->_time_frame_defs.get_end_time(frame_number));
  }
}

#endif
END_NAMESPACE_STIR


