//
// $Id$
//
/*!
  \file
  \ingroup buildblock

  \brief Declaration of class stir::AbsTimeIntervalFromDynamicData

  \author Kris Thielemans
  $Date$
  $Revision$
*/

/*
    Copyright (C) 2010- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/

#include "local/stir/AbsTimeIntervalFromDynamicData.h"
#include "stir/DynamicDiscretisedDensity.h"
#include "stir/DynamicProjData.h"
#include "stir/info.h"
#include "stir/is_null_ptr.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

const char * const
AbsTimeIntervalFromDynamicData::registered_name = "from Dynamic Data";

AbsTimeIntervalFromDynamicData::
AbsTimeIntervalFromDynamicData()
{
  set_defaults();
}

AbsTimeIntervalFromDynamicData::
AbsTimeIntervalFromDynamicData(const std::string& filename, 
                               const unsigned int start_time_frame_num,
                               const unsigned int end_time_frame_num)
    :
    _filename(filename),
    _start_time_frame_num(start_time_frame_num),
    _end_time_frame_num(end_time_frame_num)
{
  if (this->set_times() == Succeeded::no)
    error("Exiting"); // TODO should throw exception
}

Succeeded 
AbsTimeIntervalFromDynamicData::
set_times()
{
  if (this->_start_time_frame_num==0)
    {
      warning("AbsTimeIntervalFromDynamicData: need to set start_time_frame_num");
      return Succeeded::no;
    }
  if (this->_end_time_frame_num==0)
    {
      warning("AbsTimeIntervalFromDynamicData: need to set end_time_frame_num");
      return Succeeded::no;
    }

  TimeFrameDefinitions time_frame_defs;

  shared_ptr<DynamicProjData> data_sptr =
    DynamicProjData::read_from_file(this->_filename);
  
  if (!is_null_ptr(data_sptr))
    {
      this->_scan_start_time_in_secs_since_1970 = 
        data_sptr->get_start_time_in_secs_since_1970();
      time_frame_defs = data_sptr->get_time_frame_definitions();
    }
  else
    {
      info("Trying to read data as an image now.");

      shared_ptr<DynamicDiscretisedDensity> data_sptr =
        DynamicDiscretisedDensity::read_from_file(this->_filename);
      
      if (is_null_ptr(data_sptr))
        {
          return Succeeded::no;
        }

      this->_scan_start_time_in_secs_since_1970 = 
        data_sptr->get_start_time_in_secs_since_1970();
      time_frame_defs = data_sptr->get_time_frame_definitions();
    }

  this->_start_time_in_secs_since_1970 = 
    this->_scan_start_time_in_secs_since_1970 +
    time_frame_defs.get_start_time(this->_start_time_frame_num);

  this->_end_time_in_secs_since_1970 = 
    this->_scan_start_time_in_secs_since_1970 +
    time_frame_defs.get_end_time(this->_end_time_frame_num);

  return Succeeded::yes;
}


void 
AbsTimeIntervalFromDynamicData::
set_defaults()
{ 
  this->_filename ="";
  this->_start_time_in_secs_since_1970 = -1.;
  this->_start_time_frame_num = 0;
  this->_end_time_frame_num = 0;
}

void 
AbsTimeIntervalFromDynamicData::
initialise_keymap()
{ 
  parser.add_start_key("Absolute Time Interval From Dynamic Data");
  parser.add_stop_key("end Absolute Time Interval From Dynamic Data");

  parser.add_key("filename", &this->_filename);
  parser.add_key("start_frame", &this->_start_time_frame_num);
  parser.add_key("end_frame", &this->_end_time_frame_num);
}

bool
AbsTimeIntervalFromDynamicData::
post_processing()
{
  if (set_times() == Succeeded::no)
    {
      warning("AbsTimeIntervalFromDynamicData: not set properly");
      return true;
    }

  return false;
}

END_NAMESPACE_STIR
