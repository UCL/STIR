//
/*
    Copyright (C) 2006 - 2011, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup modelling
  \brief Implementation of functions of class stir::PatlakPlot

  \author Charalampos Tsoumpas
*/

#ifndef __stir_modelling_GeneralizedPatlakPlot_H__
#define __stir_modelling_GeneralizedPatlakPlot_H__

#include "stir/modelling/KineticModel.h"
#include "stir/modelling/GeneralizedPatlakMatrix.h"
#include "stir/modelling/ModelMatrix.h"
#include "stir/modelling/PlasmaData.h"
#include "stir/Succeeded.h"
#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR

//!
/*!
  \ingroup modelling
  \brief Generalized Patlak kinetic model

  Model suitable for irreversible tracers (such as FDG and FLT) AS WELL AS reversible tracers . See

  - Patlak C S, Blasberg R G, Fenstermacher J D (1985)
      <i>Graphical evaluation of blood-to-brain transfer constants from multiple-time uptake data,</i> {J Cereb Blood Flow Metab
  3(1): p. 1-7.

  - Patlak C S, Blasberg R G (1985)
    <i>Experimental and Graphical evaluation of blood-to-brain transfer constant from multiple-time uptake data:
  Generalizations,</i> J Cereb Blood Flow Metab 5: p. 584-90.


  \par Example .par file
  \verbatim
  Generalized Patlak Plot Parameters:=

  time frame definition filename := frames.txt
  starting frame := 23
  calibration factor := 9000
  blood data filename :=  blood_file.txt
  ; In seconds
  Time Shift := 0
  In total counts := 1

  end Generalized Patlak Plot Parameters:=
  \endverbatim

  \warning
  - The dynamic images will be calibrated only if the calibration factor is given.
  - The [if_total_cnt] is set to true the Dynamic Image will have the total number of
    counts while if set to false it will have the total_number_of_counts/get_duration(frame_num).
  - The dynamic images will always be in decaying counts.
  - The plasma data is assumed to be in decaying counts.

  \todo Should be derived from LinearModels, but when non-linear models will be introduced, as well.
*/
class GeneralizedPatlakPlot : public RegisteredParsingObject<GeneralizedPatlakPlot, KineticModel>
{
public:
  //! Name which will be used when parsing a GeneralizedPatlakPlot object
  static const char* const registered_name;

  GeneralizedPatlakPlot();  //!< Default constructor (calls set_defaults())
  ~GeneralizedPatlakPlot(); //!< default destructor
                            /*! \name Functions to get parameters */
                            //@{
  //! Simply gets model matrix, if it has been already stored.
  GeneralizedPatlakMatrix<2> get_model_matrix() const;
  //! Creates model matrix from plasma data (Must be already sorted in appropriate frames).
  GeneralizedPatlakMatrix<2> get_model_matrix(const PlasmaData& complete_plasma_data,
                                              const PlasmaData& plasma_frame_data,
                                              const TimeFrameDefinitions& time_frame_definitions,
                                              const unsigned int starting_frame);

  //! Simply gets model matrix, if it has been already stored.
  ModelMatrix<2> get_initialization_model_matrix() const;
  //! Creates initialization model matrix from plasma data (Must be already sorted in appropriate frames).
  ModelMatrix<2> get_initialization_model_matrix(const PlasmaData& plasma_data,
                                                 const TimeFrameDefinitions& time_frame_definitions,
                                                 const unsigned int starting_frame);

  //! Returns the frame that the GeneralizedPatlakPlot linearization is assumed to be valid.
  unsigned int get_starting_frame() const;
  //! Returns the TimeFrameDefinitions that the GeneralizedPatlakPlot linearization is assumed to be valid: ChT::Check
  TimeFrameDefinitions get_time_frame_definitions() const;
  //! Returns the number of convolution parameters for the GeneralizedPatlakPlot matrix.
  unsigned int get_num_conv_params() const;
  //!@}
  /*! \name Functions to set parameters*/
  //@{
  void set_model_matrix(GeneralizedPatlakMatrix<2> model_matrix);                   //!< Simply set model matrix
  void set_initialization_model_matrix(ModelMatrix<2> initialization_model_matrix); //!< Simply set initialization model matrix
                                                                                    //@}

  void set_Hfunction_matrix(GeneralizedPatlakMatrix<2> Hfunction_matrix); //!< Simply set Hfunction matrix
  void set_Ki_matrix(GeneralizedPatlakMatrix<2> Ki_matrix);               //!< Simply set Ki matrix

  //! Simply gets Hfunction matrix
  GeneralizedPatlakMatrix<2> get_Hfunction_matrix() const;
  GeneralizedPatlakMatrix<2> get_Ki_matrix() const;

  //! Multiplies the dynamic image with the model gradient.
  /*!  For a linear model the model gradient is the transpose of the model matrix.
    So, the dynamic image is "projected" from time domain to the parameter domain.

    \todo Should be a virtual function declared in the KineticModel class.
  */
  virtual void multiply_dynamic_image_with_model_gradient(DynamicDiscretisedDensity& impulse_response,
                                                          const DynamicDiscretisedDensity& dyn_image) const;
  //! Multiplies the dynamic image with the model gradient and add to original \c parametric_image
  /*! \todo Should be a virtual function declared in the KineticModel class.
   */
  virtual void multiply_dynamic_image_with_model_gradient_and_add_to_input(DynamicDiscretisedDensity& impulse_response,
                                                                           const DynamicDiscretisedDensity& dyn_image) const;

  virtual void get_impulse_response_from_parametric_image(DynamicDiscretisedDensity& impulse_response_image,
                                                          const GeneralizedPatlakVoxelsOnCartesianGrid& par_image) const;

  virtual void get_dynamic_image_from_impulse_response(DynamicDiscretisedDensity& dyn_image,
                                                       const DynamicDiscretisedDensity& impulse_response_image) const;

  //! Multiplies the parametric image with the model matrix to get the corresponding dynamic image.
  /*! \todo Should be a virtual function declared in the KineticModel class.
   */
  virtual void get_dynamic_image_from_parametric_image(DynamicDiscretisedDensity& dyn_image,
                                                       const GeneralizedPatlakVoxelsOnCartesianGrid& par_image) const;

  virtual void get_generalized_patlak_parameters_from_impulse_response(GeneralizedPatlakVoxelsOnCartesianGrid& par_image,
                                                                       const DynamicDiscretisedDensity& dyn_image,
                                                                       const DynamicDiscretisedDensity& impulse_response) const;

  //! Multiplies the dynamic image with the initialization kinetic model gradient.
  /*!  For a linear model the model gradient is the transpose of the model matrix.
    So, the dynamic image is "projected" from time domain to the parameter domain.

    \todo Should be a virtual function declared in the KineticModel class.
        Only used for the initialization of the Generalized Patlak Model EM update estimates
  */
  virtual void multiply_dynamic_image_with_initialization_model_gradient(GeneralizedPatlakVoxelsOnCartesianGrid& parametric_image,
                                                                         const DynamicDiscretisedDensity& dyn_image) const;

  //! Multiplies the dynamic image with the initialization kinetic model gradient and add to original \c parametric_image
  /*! \todo Should be a virtual function declared in the KineticModel class.
      //  Only used for the initialization of the Generalized Patlak Model EM update estimates
  */
  virtual void multiply_dynamic_image_with_initialization_model_gradient_and_add_to_input(
      GeneralizedPatlakVoxelsOnCartesianGrid& parametric_image, const DynamicDiscretisedDensity& dyn_image) const;

  //! Multiplies the parametric image with the initialization kinetic model matrix to get the corresponding dynamic image.
  /*! \todo Should be a virtual function declared in the KineticModel class.
      //  Only used for the initialization of the Generalized Patlak Model EM update estimates
  */
  virtual void
  get_dynamic_image_from_initialization_parametric_image(DynamicDiscretisedDensity& dyn_image,
                                                         const GeneralizedPatlakVoxelsOnCartesianGrid& par_image) const;

  virtual void estimate_nested_loop_parameters_with_model(GeneralizedPatlakVoxelsOnCartesianGrid& parametric_image,
                                                          DynamicDiscretisedDensity& dynamic_image_nested_loop_estimate,
                                                          DynamicDiscretisedDensity& dynamic_image_update_factor,
                                                          const DynamicDiscretisedDensity& dynamic_image_reference,
                                                          float minimum_nested_relative_change,
                                                          float maximum_nested_relative_change,
                                                          int num_nested_subiterations) const;

  void set_defaults();

  Succeeded set_up();

  bool _if_cardiac;                   //!< Switches between cardiac and brain data
  unsigned int _starting_frame;       //!< Starting frame to apply the model
  unsigned int _num_frames;           //!< Number of frames to apply the model
  unsigned int _conv_sample_interval; //!< Interval between convolution samples (in order to downsample the convolution sampling)
  unsigned int _num_conv_params;      //!< Number of convolution parameters
  unsigned int _last_frame_mid_time;  //!< End-time point of the last frame defined by user
  float _cal_factor;                  //!< Calibration Factor, maybe to be removed.
  float _time_shift;                  //!< Shifts the time to fit the timing of Plasma Data with the Projection Data.
  bool _in_correct_scale;    //!< Switch to scale or not the model_matrix to the correct scale, according to the appropriate scale
                             //!< factor.
  bool _in_total_cnt;        //!< Switch to choose the image values of the model to be in total counts or in mean counts.
  bool _plasma_in_total_cnt; //!< Switch to choose the plasma values of the model to be in total counts or in mean counts.
  float _kloss_lb;           //!< Lower bound for the search space of the estimated kloss parameter.
  float _kloss_ub;           //!< Upper bound for the search space of the estimated kloss parameter.
  unsigned int _kloss_num_samples;             //!< Number of samples for the search space of the estimated kloss parameter.
  std::string _blood_data_filename;            //!< Name of file in which the input function is stored
  PlasmaData _complete_plasma_data;            //!< Stores the complete plasma data before distributing/sorting into frames
  PlasmaData _plasma_frame_data;               //!< Stores the plasma data into frames for brain studies
  std::string _time_frame_definition_filename; //!< name of file to get frame definitions
  TimeFrameDefinitions _frame_defs;            //!< TimeFrameDefinitions

private:
  void create_model_matrix();                //!< Creates model matrix from private members
  void create_initialization_model_matrix(); //!< Creates initialization model matrix from private members
  void create_Hfunction_matrix();            //!< Precalculates Hfunction matrix from private members
  void create_Ki_matrix();
  void initialise_keymap();
  bool post_processing();
  mutable GeneralizedPatlakMatrix<2> _model_matrix;
  mutable ModelMatrix<2> _initialization_model_matrix;
  mutable GeneralizedPatlakMatrix<2> _Hfunction_matrix;
  mutable GeneralizedPatlakMatrix<2> _Ki_matrix;
  bool _matrix_is_stored;
  bool _initialization_matrix_is_stored;
  typedef RegisteredParsingObject<GeneralizedPatlakPlot, KineticModel> base_type;
};

END_NAMESPACE_STIR

#endif //__stir_modelling_GeneralizedPatlakPlot_H__