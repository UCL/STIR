//
//
/*!
  \file
  \ingroup buildblock

  \brief Declaration of class stir::AbsTimeIntervalWithParsing

  \author  Sanida Mustafovic and Kris Thielemans
*/

/*
    Copyright (C) 2003- 2010, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/

#include "local/stir/AbsTimeIntervalWithParsing.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

const char * const
AbsTimeIntervalWithParsing::registered_name = "secs since 1970";

static const double time_not_yet_determined=-4321;

AbsTimeIntervalWithParsing::
AbsTimeIntervalWithParsing()
{
  set_defaults();
}


void 
AbsTimeIntervalWithParsing::
set_defaults()
{ 
  _start_time_in_secs_since_1970=time_not_yet_determined;
  _end_time_in_secs_since_1970=time_not_yet_determined;
}

void 
AbsTimeIntervalWithParsing::
initialise_keymap()
{ 
  parser.add_start_key("Absolute Time Interval");
  parser.add_stop_key("end Absolute Time Interval");

  parser.add_key("start_time_in_secs_since_1970_UTC", &this->_start_time_in_secs_since_1970);
  parser.add_key("end_time_in_secs_since_1970_UTC", &this->_end_time_in_secs_since_1970);
}

bool
AbsTimeIntervalWithParsing::
post_processing()
{
  if (this->get_start_time_in_secs_since_1970() < 10000.)
    {
      warning("AbsTimeInterval: start time (%g) too small", 
	      this->get_start_time_in_secs_since_1970());
      return true;
    }
  if (this->get_duration_in_secs() <= 0.)
    {
      warning("AbsTimeInterval: duration (%g) should be > 0", 
	      this->get_duration_in_secs());
      return true;
    }

  return false;
}
END_NAMESPACE_STIR
