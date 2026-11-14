//
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
  \author Nicolas A Karakatsanis

*/

#ifndef __stir_modelling_PatlakPlot_H__
#define __stir_modelling_PatlakPlot_H__

#include "stir/modelling/KineticModel.h"
#include "stir/modelling/ModelMatrix.h"

#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR

//!
/*!
  \ingroup modelling
  \brief Patlak kinetic model

  Model suitable for irreversible tracers such as FDG and FLT. See

  - Patlak C S, Blasberg R G, Fenstermacher J D (1985)
      <i>Graphical evaluation of blood-to-brain transfer constants from multiple-time uptake data,</i> {J Cereb Blood Flow Metab
  3(1): p. 1-7.

  - Patlak C S, Blasberg R G (1985)
    <i>Experimental and Graphical evaluation of blood-to-brain transfer constant from multiple-time uptake data:
  Generalizations,</i> J Cereb Blood Flow Metab 5: p. 584-90.


  \par Example .par file
  \verbatim
  Patlak Plot Parameters:=

  time frame definition filename := frames.txt
  starting frame := 23
  calibration factor := 9000
  blood data filename :=  blood_file.txt
  ; In seconds
  Time Shift := 0
  In total counts := 1

  end Patlak Plot Parameters:=
  \endverbatim

  \warning
  - The dynamic images will be calibrated only if the calibration factor is given.
  - The [if_total_cnt] is set to true the Dynamic Image will have the total number of
    counts while if set to false it will have the total_number_of_counts/get_duration(frame_num).
  - The dynamic images will always be in decaying counts.
  - The plasma data is assumed to be in decaying counts.

  \todo Should be derived from LinearModels, but when non-linear models will be introduced, as well.
*/
class PatlakPlot : public RegisteredParsingObject<PatlakPlot, KineticModel, KineticModel>
{
private:
  typedef RegisteredParsingObject<PatlakPlot, KineticModel, KineticModel> base_type;

public:
  //! Name which will be used when parsing a PatlakPlot object
  static const char* const registered_name;

  //! Default constructor (calls set_defaults())
  PatlakPlot();

  PatlakPlot(const shared_ptr<const ExamInfo>& exam_info_sptr);

  ~PatlakPlot() override;

  /*! \name Functions to get parameters */
  //!@{
  //! Simply gets model matrix, if it has been already stored.
  ModelMatrix<2> get_model_matrix() const;
  //! Creates model matrix from plasma data (Must be already sorted in appropriate frames).
  ModelMatrix<2> get_model_matrix(const PlasmaData& plasma_data,
                                  const TimeFrameDefinitions& time_frame_definitions,
                                  const unsigned int starting_frame);

  //!@}
  /*! \name Functions to set parameters*/
  //!@{
  void set_model_matrix(ModelMatrix<2> model_matrix); //!< Simply set model matrix
  //!@}

  //! Multiplies the dynamic image with the model gradient.
  /*!  For a linear model the model gradient is the transpose of the model matrix.
    So, the dynamic image is "projected" from time domain to the parameter domain.

    \todo Should be a virtual function declared in the KineticModel class.
  */
  virtual void multiply_dynamic_image_with_model_gradient(ParametricVoxelsOnCartesianGrid& parametric_image,
                                                          const DynamicDiscretisedDensity& dyn_image) const;
  //! Multiplies the dynamic image with the model gradient and add to original \c parametric_image
  /*! \todo Should be a virtual function declared in the KineticModel class.
   */
  virtual void multiply_dynamic_image_with_model_gradient_and_add_to_input(ParametricVoxelsOnCartesianGrid& parametric_image,
                                                                           const DynamicDiscretisedDensity& dyn_image) const;

  //! Multiplies the parametric image with the model matrix to get the corresponding dynamic image.
  /*! \todo Should be a virtual function declared in the KineticModel class.
   */
  virtual void get_dynamic_image_from_parametric_image(DynamicDiscretisedDensity& dyn_image,
                                                       const ParametricVoxelsOnCartesianGrid& par_image) const;

  //! Multiplies the dynamic image with the initialization kinetic model gradient.
  /*!  For a linear model the model gradient is the transpose of the model matrix.
    So, the dynamic image is "projected" from time domain to the parameter domain.

    \todo Should be a virtual function declared in the KineticModel class.
        Only intended for the initialization of the Generalized Patlak Model EM update estimates
        Currently not used but retained for future potential usage.
    The initialization of generalized Patlak nested estimates is performed by GeneralizedPatlakPlot equivalent method
  */
  virtual void multiply_dynamic_image_with_initialization_model_gradient(Parametric3VoxelsOnCartesianGrid& parametric_image,
                                                                         const DynamicDiscretisedDensity& dyn_image) const;

  //! Multiplies the dynamic image with the initialization kinetic model gradient and add to original \c parametric_image
  /*! \todo Should be a virtual function declared in the KineticModel class.
      Only intended for the initialization of the Generalized Patlak Model EM update estimates
          Currently not used but retained for future potential usage.
      The initialization of generalized Patlak nested estimates is performed by GeneralizedPatlakPlot equivalent method
  */
  virtual void
  multiply_dynamic_image_with_initialization_model_gradient_and_add_to_input(Parametric3VoxelsOnCartesianGrid& parametric_image,
                                                                             const DynamicDiscretisedDensity& dyn_image) const;

  //! Multiplies the parametric image with the initialization kinetic model matrix to get the corresponding dynamic image.
  /*! \todo Should be a virtual function declared in the KineticModel class.
          Only intended for the initialization of the Generalized Patlak Model EM update estimates
          Currently not used but retained for future potential usage.
      The initialization of generalized Patlak nested estimates is performed by GeneralizedPatlakPlot equivalent method
  */
  virtual void get_dynamic_image_from_initialization_parametric_image(DynamicDiscretisedDensity& dyn_image,
                                                                      const Parametric3VoxelsOnCartesianGrid& par_image) const;

  //! This is the common method used to estimate the parametric images from the dynamic images.
  /*! \todo There is currently no check if the time frame definitions from \a dyn_image are
    the same as the ones encoded in the model.
  */
  void apply_linear_regression(ParametricVoxelsOnCartesianGrid& par_image, const DynamicDiscretisedDensity& dyn_image) const;

  void set_defaults() override;

  Succeeded set_up() override;

private:
  //! Creates model matrix from private members
  void create_model_matrix() override;

  void initialise_keymap() override;
  bool post_processing() override;
  mutable ModelMatrix<2> _model_matrix;
};

END_NAMESPACE_STIR

#endif //__stir_modelling_PatlakPlot_H__
