//
// $Id$
//
/*!
  \file
  \ingroup motion

  \brief Declaration of class stir::AbsTimeIntervalWithParsing

  \author  Sanida Mustafovic and Kris Thielemans
  $Date$
  $Revision$
*/

/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/

#include "local/stir/AbsTimeIntervalWithParsing.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

const char * const
AbsTimeIntervalWithParsing::registered_name = "secs since 1970";

static const std::time_t time_not_yet_determined=-4321;

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

  parser.add_key("start_time_in_secs_since_1970_UTC", &_start_time_in_secs_since_1970_for_parsing);
  parser.add_key("end_time_in_secs_since_1970_UTC", &_end_time_in_secs_since_1970_for_parsing);
}

bool
AbsTimeIntervalWithParsing::
post_processing()
{
  _start_time_in_secs_since_1970 =
    static_cast<std::time_t>(_start_time_in_secs_since_1970_for_parsing);
  _end_time_in_secs_since_1970 =
    static_cast<std::time_t>(_end_time_in_secs_since_1970_for_parsing);

  if (this->get_start_time_in_secs_since_1970() < 10000)
    {
      warning("AbsTimeInterval: start time (%d) too small", 
	      this->get_start_time_in_secs_since_1970());
      return true;
    }
  if (this->get_duration_in_secs() <= 0)
    {
      warning("AbsTimeInterval: duration (%d) should be > 0", 
	      this->get_duration_in_secs());
      return true;
    }

  return false;
}
END_NAMESPACE_STIR
