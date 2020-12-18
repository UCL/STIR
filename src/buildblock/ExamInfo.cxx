/*
    Copyright (C) 2013, 2014, 2018, 2020, University College London
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
  \ingroup buildblock
  \brief implementation of stir::ExamInfo

  \author Kris Thielemans
*/
#include "stir/ExamInfo.h"
#include "stir/date_time_functions.h"

#ifdef BOOST_NO_STRINGSTREAM
#include <strstream.h>
#else
#include <sstream>
#endif

START_NAMESPACE_STIR

std::string
ExamInfo::parameter_info() const
{
#ifdef BOOST_NO_STRINGSTREAM
  // dangerous for out-of-range, but 'old-style' ostrstream seems to need this
  char str[30000];
  ostrstream s(str, 30000);
#else
  std::ostringstream s;
#endif
  s << "Modality: " << this->imaging_modality.get_name() << '\n';
  s << "Patient position: " << this->patient_position.get_position_as_string() << '\n';
  s << "Scan start time: " << this->start_time_in_secs_since_1970 << '\n';
  if (this->start_time_in_secs_since_1970>0)
    {
      DateTimeStrings time = secs_since_Unix_epoch_to_Interfile_datetime(this->start_time_in_secs_since_1970);
      s << "   which is " << time.date << " " << time.time << '\n';
    }
  if (this->time_frame_definitions.get_num_time_frames() == 1)
    {
      s << "Time frame start - end (duration), all in secs: "
        << this->time_frame_definitions.get_start_time(1)
        << " - "
        << this->time_frame_definitions.get_end_time(1)
        << " ("
        << this->time_frame_definitions.get_duration(1)
        << ")\n";
    }
  s << "number of energy windows:=1\n"
    << "energy window lower level[1] := "
    << this->get_low_energy_thres() << '\n'
    << "energy window upper level[1] := "
    << this->get_high_energy_thres() << '\n';
        
  return s.str();
}


END_NAMESPACE_STIR
