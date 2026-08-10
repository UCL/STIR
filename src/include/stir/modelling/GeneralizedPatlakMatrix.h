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
  \brief Declaration of class stir::GeneralizedPatlakMatrix
  \author Nicolas A Karakatsanis

*/

#ifndef __stir_modelling_GeneralizedPatlakMatrix_H__
#define __stir_modelling_GeneralizedPatlakMatrix_H__

#include "stir/Array.h"
#include "stir/BasicCoordinate.h"
#include "stir/VectorWithOffset.h"
#include "stir/DynamicDiscretisedDensity.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/Succeeded.h"
#include <fstream>
#include <iostream>

START_NAMESPACE_STIR
//! A helper class to store the model matrix for a linear kinetic model
/*! \ingroup modelling
 */
template <int num_conv_points>
class GeneralizedPatlakMatrix
{
public:
  inline GeneralizedPatlakMatrix(); //!< default constructor

  inline ~GeneralizedPatlakMatrix(); //!< default destructor

  /*! Implementation to read the model matrix from a text file
    \warning In this way the information about the calibration _is_uncalibrated and the counts _is_converted is not passed.
  */
  inline void read_from_file(const std::string input_string, int num_conv_params);

  //! Implementation to write the model matrix to a text file
  inline Succeeded write_to_file(const std::string output_string, int num_conv_params);

  //! \name Functions to get parameters @{
  inline Array<2, float> get_model_array() const;
  inline const VectorWithOffset<float> get_model_array_sum() const;
  inline VectorWithOffset<float> get_time_vector() const;
  //!@}
  //! \name Functions to set parameters @{
  inline void set_model_array(const Array<2, float>& model_array);

  inline void set_Hfunction_array(const Array<2, float>& Hfunction_array);
  inline void set_Ki_array(const Array<2, float>& Hfunction_array);

  inline void set_conv_sample_interval(const unsigned int conv_sampling_interval);

  inline void set_prefetched_sampling(const float kloss_start, const float kloss_end, const float kloss_nsamples);

  inline Array<2, float> get_Hfunction_array() const;
  inline Array<2, float> get_Ki_array() const;

  inline void estimate_inverse_Hfunction(float& kloss_estimate, float& Hfunction_val) const;

  inline void estimate_denominator_Kifunction(float& Ki_denominator, float& kloss_val) const;

  inline void set_time_vector(const VectorWithOffset<float>& time_vector);
  //! Function to set _is_calibrated boolean true or false
  inline void set_if_uncalibrated(const bool is_uncalibrated);
  inline void set_if_in_correct_scale(const bool in_correct_scale);
  inline void set_matrix_in_total_frame_counts(const bool is_converted_to_total_counts);
  //!@}

  //! Function to give the threshold_value to the all elements of the model_array which lower value than the threshold_value.
  inline void threshold_model_array(const float threshold_value);

  /*! Function to divide with the calibration factor the model array.
    Calibrated ModelMatrix means that the counts are in kBq/ml, while uncalibrated means that it will be to the same units as the
    reconstructed images.
   */
  inline void uncalibrate(const float cal_factor);

  /*! Function to multiply with the scale factor the model array.
    Scaled ModelMatrix means that the counts are already scaled to the correct, while not scaled means that it needs to be scaled.
   */
  inline void scale_model_matrix(const float scale_factor);

  /*! Multiply with the duration to convert the count rate to total counts in the time frame.
    Converted ModelMatrix means that it is in total counts in respect to the time_frame_duration,
    while not converted sets the _is_converted to false and means that it will be in "mean count rate".
   */
  inline void convert_to_total_frame_counts(const TimeFrameDefinitions& time_frame_definitions);

  /*! Multiplications of the model with the dynamic or the parametric images.
    /todo Maybe it will be better to lie in a linear models class.
  */
  //@{
  //! multiply (transpose) model-matrix with dynamic image and add result to original \c parametric_image
  inline void multiply_dynamic_image_with_model_and_add_to_input(DynamicDiscretisedDensity& impulse_response_image,
                                                                 const DynamicDiscretisedDensity& dynamic_image,
                                                                 int num_conv_params) const;
  //! multiply (transpose) model-matrix with dynamic image (overwriting original content of \c parametric_image)
  /*! \todo current implementation first fills first argument with 0 and then calls
   multiply_dynamic_image_with_model_and_add_to_input(). This is somewhat inefficient.
  */
  inline void multiply_dynamic_image_with_model(DynamicDiscretisedDensity& impulse_response_image,
                                                const DynamicDiscretisedDensity& dynamic_image,
                                                int num_conv_params) const;

  inline void synthesize_impulse_response_from_parametric_image(DynamicDiscretisedDensity& impulse_response_image,
                                                                const GeneralizedPatlakVoxelsOnCartesianGrid& parametric_image,
                                                                int num_conv_params) const;

  inline void multiply_impulse_response_with_model_and_add_to_input(DynamicDiscretisedDensity& dynamic_image,
                                                                    const DynamicDiscretisedDensity& impulse_response_image,
                                                                    int num_conv_params) const;

  inline void multiply_impulse_response_with_model(DynamicDiscretisedDensity& dynamic_image,
                                                   const DynamicDiscretisedDensity& impulse_response_image,
                                                   int num_conv_params) const;

  //! multiply model-matrix with parametric image and add result to original \c dynamic_image
  inline void
  multiply_parametric_image_with_model_and_add_to_input(DynamicDiscretisedDensity& dynamic_image,
                                                        const GeneralizedPatlakVoxelsOnCartesianGrid& parametric_image,
                                                        int num_conv_params) const;
  //! multiply model-matrix with parametric image (overwriting original content of \c dynamic_image)
  /*! \todo current implementation first fills first argument with 0 and then calls
   multiply_dynamic_image_with_model_and_add_to_input(). This is somewhat inefficient.
  */
  inline void multiply_parametric_image_with_model(DynamicDiscretisedDensity& dynamic_image,
                                                   const GeneralizedPatlakVoxelsOnCartesianGrid& parametric_image,
                                                   int num_conv_params) const;

  inline void estimate_generalized_patlak_parameters_with_impulse_response_and_add_to_input(
      GeneralizedPatlakVoxelsOnCartesianGrid& parametric_image,
      const DynamicDiscretisedDensity& impulse_response_image,
      int num_conv_params) const;

  inline void
  estimate_generalized_patlak_parameters_with_impulse_response(GeneralizedPatlakVoxelsOnCartesianGrid& parametric_image,
                                                               const DynamicDiscretisedDensity& impulse_response_image,
                                                               int num_conv_params) const;

  inline void normalise_parametric_image_with_model_sum(GeneralizedPatlakVoxelsOnCartesianGrid& parametric_image_out,
                                                        const GeneralizedPatlakVoxelsOnCartesianGrid& parametric_image,
                                                        int num_conv_params) const;

  inline void estimate_nested_loop_parameters_with_model(GeneralizedPatlakVoxelsOnCartesianGrid& parametric_image,
                                                         DynamicDiscretisedDensity& dynamic_image_nested_loop_estimate,
                                                         DynamicDiscretisedDensity& dynamic_image_update_factor,
                                                         const DynamicDiscretisedDensity& dynamic_image_reference,
                                                         int num_nested_subiterations,
                                                         float min_nested_rel_change,
                                                         float max_nested_rel_change,
                                                         int num_conv_params) const;

  //@}
private:
  //! At the moment it has the form of _model_array[param_num][frame_num].
  Array<2, float> _model_array;
  Array<2, float> _Hfunction_array;
  Array<2, float> _Ki_array;
  VectorWithOffset<float> _time_vector;
  bool _is_uncalibrated;
  bool _in_correct_scale;
  bool _is_converted_to_total_counts;
  unsigned int _conv_sampling_interval;
  float _kloss_start;
  float _kloss_end;
  unsigned int _kloss_nsamples;
};

END_NAMESPACE_STIR

#include "stir/modelling/GeneralizedPatlakMatrix.inl"

#endif //__stir_modelling_GeneralizedPatlakMatrix_H__