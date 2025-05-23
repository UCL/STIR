//
//
/*
    Copyright (C) 2003- 2007, Hammersmith Imanet Ltd
    Copyright (C) 2021, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup singles_buildblock
  \brief Implementation of stir::SinglesRatesForTimeFrames

  \author Kris Thielemans
  \author Timothy Borgeaud
*/

#include "stir/data/SinglesRatesForTimeFrames.h"
//#include "stir/IndexRange2D.h"

START_NAMESPACE_STIR

SinglesRatesForTimeFrames::
SinglesRatesForTimeFrames()
{}


float 
SinglesRatesForTimeFrames::
get_singles(const int singles_bin_index,
            const unsigned int time_frame_num) const
{ 
  return(this->_singles[time_frame_num][singles_bin_index]);
}

void 
SinglesRatesForTimeFrames::
set_singles(const int singles_bin_index,
            const unsigned time_frame_num,
            const float new_singles)
{
  this->_singles[time_frame_num][singles_bin_index] = new_singles;
}


float
SinglesRatesForTimeFrames::
get_singles(const int singles_bin_index,
            const double start_time,
            const double end_time) const
{
  const unsigned frame_number = 
    this->_time_frame_defs.get_time_frame_num(start_time, end_time);
  if (frame_number == 0)
    return -1.F;
  else
  return(get_singles(singles_bin_index, frame_number));
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


