//
// $Id$
//
/*!
  \file
  \ingroup motion

  \brief Declaration of class stir::AbsTimeIntervalFromECAT7ACF

  \author  Sanida Mustafovic and Kris Thielemans
  $Date$
  $Revision$
*/

/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/

#include "local/stir/AbsTimeIntervalFromECAT7ACF.h"
#include "stir/IO/stir_ecat7.h"
#include "stir/round.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

const char * const
AbsTimeIntervalFromECAT7ACF::registered_name = "from ECAT7 ACF";

AbsTimeIntervalFromECAT7ACF::
AbsTimeIntervalFromECAT7ACF()
{
  set_defaults();
}

AbsTimeIntervalFromECAT7ACF::
AbsTimeIntervalFromECAT7ACF(const std::string& filename, double duration_in_secs)
    :
    _attenuation_filename(filename),
    _transmission_duration(duration_in_secs)    
{
  if (set_times() == Succeeded::no)
    error("Exiting"); // TODO should throw exception
}

Succeeded 
AbsTimeIntervalFromECAT7ACF::
set_times()
{
#ifdef HAVE_LLN_MATRIX
  if (_transmission_duration<=0)
    {
      warning("AbsTimeIntervalFromECAT7ACF: duration should be > 0 but is %g%.",
	      _transmission_duration);
      return Succeeded::no;
    }
  MatrixFile* attn_file = matrix_open(_attenuation_filename.c_str(), MAT_READ_ONLY, AttenCor );
  if (attn_file==NULL)
    {
      warning("Error opening attenuation file '%s'", _attenuation_filename.c_str());
      return Succeeded::no;
    }

  _start_time_in_secs_since_1970 = attn_file->mhptr->scan_start_time;
  _end_time_in_secs_since_1970 = 
    _start_time_in_secs_since_1970 + round(floor(_transmission_duration));

  matrix_close(attn_file);

  return Succeeded::yes;
#else
    warning("Error opening attenuation file %s: compiled without ECAT7 support.", 
	    attenuation_filename.c_str());
    return Succeeded::no;
#endif
}


void 
AbsTimeIntervalFromECAT7ACF::
set_defaults()
{ 
  _transmission_duration = -1;
  _attenuation_filename ="";
}

void 
AbsTimeIntervalFromECAT7ACF::
initialise_keymap()
{ 
  parser.add_start_key("Absolute Time Interval From ECAT7 ACF");
  parser.add_stop_key("end Absolute Time Interval From ECAT7 ACF");

  parser.add_key("attenuation_filename", &_attenuation_filename);
  parser.add_key("transmission_duration", &_transmission_duration);
}

bool
AbsTimeIntervalFromECAT7ACF::
post_processing()
{
  if (set_times() == Succeeded::no)
    {
      warning("AbsTimeIntervalFromECAT7ACF: not set properly");
      return true;
    }

  return false;
}

END_NAMESPACE_STIR
