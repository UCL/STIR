/*
    Copyright (C) 2021 National Physical Laboratory
    Copyright (C) 2013, 2014, 2018, 2020-2021 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup buildblock
  \brief implementation of stir::ExamInfo

  \author Kris Thielemans
  \author Daniel Deidda
*/
#include "stir/ExamInfo.h"
#include "stir/date_time_functions.h"
#include <iomanip>

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
  s << "Calibration Factor: " << std::fixed << std::setprecision(12) << this->calibration_factor << std::setprecision(5) << '\n';
  s << "Radionuclide: " << this->radionuclide << '\n';
  s << "Patient position: " << this->patient_position.get_position_as_string() << '\n';
  s << "Scan start time: " << std::fixed << std::setprecision(1) <<this->start_time_in_secs_since_1970 <<'\n'; // reset for further floats
 
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

bool 
ExamInfo::operator == (const ExamInfo &p1) const {      
    return  abs(this->up_energy_thres - p1.up_energy_thres )<=1 && /* keV*/
            abs(this->low_energy_thres - p1.low_energy_thres) <=1 &&/* keV*/
            this->radionuclide==p1.radionuclide &&
            this->time_frame_definitions==p1.time_frame_definitions &&
//              this->branching_ratio==p1.branching_ratio &&
            ((this->calibration_factor<=0 && p1.calibration_factor<=0) || 
             abs(this->calibration_factor/p1.calibration_factor -1.)<=1E-3) &&
            this->imaging_modality==p1.imaging_modality &&
            this->patient_position==p1.patient_position &&
            abs(this->start_time_in_secs_since_1970 - p1.start_time_in_secs_since_1970)<=.5;/* sec */ }

END_NAMESPACE_STIR
