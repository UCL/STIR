/*
    Copyright (C) 2013, 2018, University College London
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
  \brief  This file declares the class stir::ExamInfo
  \author Kris Thielemans
*/


#ifndef __stir_ExamInfo_H__
#define __stir_ExamInfo_H__

#include "stir/PatientPosition.h"
#include "stir/TimeFrameDefinitions.h"
#include "stir/ImagingModality.h"
#include "stir/shared_ptr.h"

START_NAMESPACE_STIR


/*!
  \brief a class for storing information about 1 exam (or scan)
  \ingroup buildblock
  \todo this is very incomplete at the moment. Things like bed positions, gating, isotopes etc etc are all missing

  \todo This should be an abtract registered object, in oreder to serve as a complete
  base function for every input data type.
  */
class ExamInfo
{

public :

  //! Default constructor
  /*! Most fields take there default values (much might be invalid).
     \a start_time_in_secs_since_1970 is set to zero to 
     indicate that it is not initialised.
  */

  ExamInfo()
    : start_time_in_secs_since_1970(0.)
    {
      low_energy_thres = -1.f;
      up_energy_thres = -1.f;
  }

  std::string originating_system;
  
  ImagingModality imaging_modality;

  PatientPosition patient_position;

  TimeFrameDefinitions time_frame_definitions;

  const TimeFrameDefinitions& get_time_frame_definitions() const
  { return time_frame_definitions; }
  TimeFrameDefinitions& get_time_frame_definitions()
  { return time_frame_definitions; }

  double start_time_in_secs_since_1970;

  //! \name Functions that return info related on the acquisition settings
  //@{
  //! Get the low energy boundary
  inline float get_low_energy_thres() const;
  //! Get the high energy boundary
  inline float get_high_energy_thres() const;
  //@}

  //! \name Functions that set values related on the acquisition settings
  //@{
  //! Set the low energy boundary
  inline void set_low_energy_thres(float new_val);
  //! Set the high energy boundary
  inline void set_high_energy_thres(float new_val);
  //@}

  void set_time_frame_definitions(const TimeFrameDefinitions& new_time_frame_definitions)
    {
      time_frame_definitions = new_time_frame_definitions;
    }

  //! Clone and create shared_ptr of the copy
  shared_ptr<ExamInfo> create_shared_clone()
  {
      return shared_ptr<ExamInfo>(new ExamInfo(*this));
  }

  private:
     //!
  //! \brief low_energy_thres
  //! \author Nikos Efthimiou
  //! \details This is the value of low energy threshold of the energy window.
  //! The units are keV
  //! This parameter was initially introduced for scatter simulation.
  //! If scatter simulation is not needed, can default to -1
  float low_energy_thres;

  //!
  //! \brief up_energy_thres
  //! \author Nikos Efthimiou
  //! \details This is the value of high energy threshold of the energy window
  //! The units are keV
  //! This parameter was initially introduced for scatter simulation
  //! If scatter simulation is not needed, can default to -1
  float up_energy_thres;
};

END_NAMESPACE_STIR

#include "stir/ExamInfo.inl"

#endif // __stir_ExamInfo_H__
