/*
    Copyright (C) 2021 National Physical Laboratory
    Copyright (C) 2013, 2018, 2020-2021 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*! 
  \file
  \ingroup buildblock
  \brief  This file declares the class stir::ExamInfo
  \author Kris Thielemans
  \author Nikos Efthimiou
  \author Daniel Deidda
*/


#ifndef __stir_ExamInfo_H__
#define __stir_ExamInfo_H__

#include "stir/PatientPosition.h"
#include "stir/TimeFrameDefinitions.h"
#include "stir/ImagingModality.h"
#include "stir/Radionuclide.h"
#include "stir/shared_ptr.h"

START_NAMESPACE_STIR


/*!
  \brief a class for storing information about 1 exam (or scan)
  \ingroup buildblock
  \todo this is very incomplete at the moment. Things like bed positions, gating, isotopes etc etc are all missing

  \todo This should be an abtract registered object, in order to serve as a complete
  base function for every input data type.
  
  */
class ExamInfo
{

public :

  //! Default constructor
  /*! Most fields take their default values (which might be invalid).
     \a start_time_in_secs_since_1970 is set to zero, energy window info to -1, to
     indicate that it is not initialised.
  */

  ExamInfo()
    : start_time_in_secs_since_1970(0.),
    calibration_factor(-1.F)
    {

      num_windows = 1;
      low_energy_thres.resize(1);
      up_energy_thres.resize(1);
      en_win_pair.resize(2);
      low_energy_thres[0]=-1.F;
      up_energy_thres[0]=-1.F;
      en_win_pair[0]=1;
      en_win_pair[1]=1;

   }


  std::string originating_system;
    
  ImagingModality imaging_modality;

  PatientPosition patient_position;

  TimeFrameDefinitions time_frame_definitions;
  
  Radionuclide radionuclide;
  
//  double branching_ratio;

  const TimeFrameDefinitions& get_time_frame_definitions() const
  { return time_frame_definitions; }
  TimeFrameDefinitions& get_time_frame_definitions()
  { return time_frame_definitions; }

  double start_time_in_secs_since_1970;

  //! \name Functions that return info related on the acquisition settings
  //@{
  //! Get the low energy boundary
  inline float  get_low_energy_thres(int en_window = 0) const;
  //! Get the high energy boundary
  inline float  get_high_energy_thres(int en_window = 0) const;

  //! Get the number of energy windows
  inline int  get_num_energy_windows() const;
  inline std::pair<int,int> get_energy_window_pair() const;
  //! Get the calibration factor
  inline  float get_calibration_factor() const;
  //! Get the radionuclide name
  inline Radionuclide get_radionuclide() const;
  //@}


  //! \name Functions that set values related on the acquisition settings
  //@{
  //! Set the low energy boundary
  inline void set_low_energy_thres(float new_val,int en_window = 0);
  //! Set the high energy boundary
  inline void set_high_energy_thres(float new_val,int en_window = 0);
  //! Set the low energy boundary as a vector (currently only used in listmode code)
  inline void set_low_energy_thres_vect(std::vector<float> new_val);
  //! Set the high energy boundary as a vector (currently only used in listmode code)
  inline void set_high_energy_thres_vect(std::vector<float> new_val);
  //! Set the number of energy windows
  inline void set_num_energy_windows(int n_win);
  //! Set the energy window pair
  inline void set_energy_window_pair(std::vector<int> val);

  //! Set the Calibration factor
  inline void set_calibration_factor(const float cal_val);
  //! Set the radionuclide
  inline void set_radionuclide(const Radionuclide& arg);
  //! Copy energy information from another ExamInfo
  inline void set_energy_information_from(const ExamInfo&);
  //@}

  inline bool has_energy_information() const
  {
    //TODO ADD LOOP: for (int i=0; i< num_windows; i++)
    return (low_energy_thres[0] > 0.f)&&(up_energy_thres[0] > 0.f);
  }

  //! Standard trick for a 'virtual copy-constructor'
  inline ExamInfo* clone() const;
  //! Like clone() but return a shared_ptr
  inline shared_ptr<ExamInfo> create_shared_clone() const;

  void set_time_frame_definitions(const TimeFrameDefinitions& new_time_frame_definitions)
    {
      time_frame_definitions = new_time_frame_definitions;
    }

  //!  Warning: the operator == does not check that originating system is consistent!
  bool operator == (const ExamInfo &p1) const ;
  
  //! Clone and create shared_ptr of the copy
  shared_ptr<ExamInfo> create_shared_clone()
  {
      return shared_ptr<ExamInfo>(new ExamInfo(*this));
  }

  //! Return a string with info on parameters
  /*! the returned string is not intended for parsing. */
  std::string parameter_info() const;

protected:
  
  float calibration_factor;
  private:
  //!
  //! \brief num_windows
  //! \author Ludovica Brusaferri
  //! \details This is the value of the number of enegry windows
  //! This parameter was initially introduced for scatter simulation.
  //! If scatter simulation is not needed, can default to 1
  int num_windows;

  //!
  //! \brief low_energy_thres
  //! \author Nikos Efthimiou
  //! \details This is the value of low energy threshold of the energy window.
  //! The units are keV
  //! This parameter was initially introduced for scatter simulation.
  //! If scatter simulation is not needed, can default to -1
  std::vector<float> low_energy_thres;

  //!
  //! \brief up_energy_thres
  //! \author Nikos Efthimiou
  //! \details This is the value of high energy threshold of the energy window
  //! The units are keV
  //! This parameter was initially introduced for scatter simulation
  //! If scatter simulation is not needed, can default to -1
  std::vector<float> up_energy_thres;

  //!
  //! \brief en_win_pair
  //! \author Ludovica Brusaferri
  //! \details This is the value of the energy window pair for a certain sinogram
  //! Can default to {1,1}
  std::vector<int> en_win_pair;

};

END_NAMESPACE_STIR

#include "stir/ExamInfo.inl"

#endif // __stir_ExamInfo_H__
