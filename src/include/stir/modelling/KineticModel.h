//
//
/*
    Copyright (C) 2006 - 2009, Hammersmith Imanet Ltd
    Copyright (C) 2026, University Medical Center Groningen
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup modelling

  \brief Definition of class stir::KineticModel

  \author Charalampos Tsoumpas
  \author Nikos Efthimiou
*/

#ifndef __stir_modelling_KineticModel_H__
#define __stir_modelling_KineticModel_H__

#include "stir/RegisteredObject.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/TimeFrameDefinitions.h"
#include "stir/modelling/PlasmaData.h"
#include "stir/Succeeded.h"
#include "stir/ExamInfo.h"

#include "stir/shared_ptr.h"

START_NAMESPACE_STIR

/*!
  \brief base class for all kinetic models
  \ingroup modelling

  At present very basic. It just provides the parsing mechanism.
*/
class KineticModel : public RegisteredObject<KineticModel>
{

public:
  static const char* const registered_name;
  //! default constructor
  KineticModel() { _already_setup = false; };

  KineticModel(const ExamInfo& _exam_info_arg) { _already_setup = false; };
  //! default destructor
  virtual ~KineticModel(){};

  //  virtual float get_compartmental_activity_at_time(const int param_num, const int sample_num) const;
  //  virtual float get_total_activity_at_time(const int sample_num) const;

  virtual Succeeded set_up();

  /*! \name Functions to get parameters */
  //@{

  //! Returns the frame that the GeneralizedPatlakPlot linearization is assumed to be valid.
  inline unsigned int get_starting_frame() const;
  //! Returns the TimeFrameDefinitions that the GeneralizedPatlakPlot linearization is assumed to be valid: ChT::Check
  inline const TimeFrameDefinitions& get_time_frame_definitions() const;
  //! Returns the number of the last frame available.
  inline unsigned int get_ending_frame() const;

  inline const PlasmaData& get_plasma_data() const;

  inline float get_calibration_factor() const;

  inline const shared_ptr<const ExamInfo> get_exam_info_sptr() const;

  inline int get_frame_reference_time() const;
  //!@}

  /*! \name Functions to set parameters */
  //@{
  inline void set_plasma_data(PlasmaData& arg);

  inline void set_time_frame_definitions(TimeFrameDefinitions& arg);

  inline void set_starting_frame(unsigned int arg);

  inline void set_calibration_factor(float arg);

  inline void set_exam_info(const shared_ptr<const ExamInfo>& exam_info_sptr);

  inline void set_radionuclide(const Radionuclide _radionuclide);

  inline void set_frame_reference_time(int _arg);

  //!@}

protected:
  virtual void initialise_keymap();
  virtual void set_defaults();
  virtual bool post_processing();
  //!
  virtual void create_model_matrix();
  //! Switches between cardiac and brain data
  bool _if_cardiac;
  //! Calibration Factor, maybe to be removed.
  float _cal_factor;
  //! Shifts the time to fit the timing of Plasma Data with the Projection Data.
  float _time_shift;
  //! Switch to scale or not the model_matrix to the correct scale, according to the appropriate scale
  bool _in_correct_scale;
  //! Switch to choose the image values of the model to be in total counts or in mean counts.
  bool _in_total_cnt;
  //! Switch to choose the plasma values of the model to be in total counts or in mean counts.
  bool _plasma_in_total_cnt;
  //! Name of file in which the input function is stored
  std::string _blood_data_filename;
  //! Stores the plasma data into frames for brain studies
  PlasmaData _plasma_frame_data;
  //! name of file to get frame definitions
  std::string _time_frame_definition_filename;

  bool _matrix_is_stored;

private:
  //! All setters will set this to false, reminding you call set_up()
  bool _already_setup;
  //! TimeFrameDefinitions
  TimeFrameDefinitions _frame_defs;
  //! Starting frame to apply the model
  unsigned int _starting_frame;

  std::shared_ptr<const ExamInfo> _exam_info_sptr;

  //! end_of_frame = 0
  //! mid_of_frame = 1
  int _frame_reference_time;
};

END_NAMESPACE_STIR
#include "stir/modelling/KineticModel.inl"

#endif //__stir_modelling_KineticModel_H__
