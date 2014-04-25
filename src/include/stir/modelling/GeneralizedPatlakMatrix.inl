//
// $Id: GeneralizedPatlakMatrix.inl,v 1.4 2011-01-12 18:27:12 kris Exp $
//
/*
    Copyright (C) 2006 - $Date: 2011-01-12 18:27:12 $, Hammersmith Imanet Ltd
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

  \file
  \ingroup modelling

  \brief Implementations of inline functions of class stir::GeneralizedPatlakMatrix

  \author Nicolas A Karakatsanis

  $Date: 2011-01-12 18:27:12 $
  $Revision: 1.4 $
*/

#include <algorithm>
#include <math.h>
#include "stir/numerics/divide.h"
#include "stir/thresholding.h"

using std::cerr;
using std::endl;

START_NAMESPACE_STIR

const float small_num = 0.000001F;

//! default constructor
template <int num_conv_points>
GeneralizedPatlakMatrix<num_conv_points>::GeneralizedPatlakMatrix()
{
  // Calibrated ModelMatrix means that the counts are in kBq/ml, while uncalibrated means that it will be to the same units as the
  // reconstructed images.
  this->_is_uncalibrated = false;
  // Converted ModelMatrix means that it is in total counts in respect to the time_frame_duration, while false means that it will
  // be in mean count rate.
  this->_in_correct_scale = false;
  this->_is_converted_to_total_counts = false;
}

//! default destructor
template <int num_conv_points>
GeneralizedPatlakMatrix<num_conv_points>::~GeneralizedPatlakMatrix()
{}

//! Implementation to read the model matrix
template <int num_conv_points>
void
GeneralizedPatlakMatrix<num_conv_points>::read_from_file(const std::string input_string, int num_conv_params)
{
  std::ifstream data_stream(input_string.c_str());
  unsigned int starting_frame, last_frame;
  if (!data_stream)
    error("cannot read model matrix from file.\n");
  else
    {
      data_stream >> starting_frame;
      data_stream >> last_frame;
    }

  BasicCoordinate<2, int> min_range;
  BasicCoordinate<2, int> max_range;
  min_range[1] = 1;
  min_range[2] = starting_frame;
  max_range[1] = num_conv_params;
  max_range[2] = last_frame;
  IndexRange<2> data_range(min_range, max_range);
  Array<2, float> input_array(data_range);
  while (true)
    {
      for (unsigned int frame_num = starting_frame; frame_num <= last_frame; ++frame_num)
        for (int param_num = 1; param_num <= num_conv_params; ++param_num)
          data_stream >> input_array[param_num][frame_num];
      if (!data_stream)
        break;
    }
  this->_model_array = input_array; // I do not pass info if it is calibrated and if it includes time frame_duration, yet.
}

//! Implementation to write the model matrix
template <int num_conv_points>
Succeeded
GeneralizedPatlakMatrix<num_conv_points>::write_to_file(const std::string output_string, int num_conv_params)
{

  BasicCoordinate<2, int> model_array_min, model_array_max;
  if (!(this->_model_array).get_regular_range(model_array_min, model_array_max))
    error("Model array has not regular range");
  unsigned int starting_frame = model_array_min[2], last_frame = model_array_max[2];

  std::ofstream data_stream(output_string.c_str(), std::ios::out);
  if (!data_stream)
    {
      warning("GeneralizedPatlakMatrix::write_to_file: error opening output file %s\n", output_string.c_str());
      return Succeeded::no;
    }
  else
    {
      data_stream << starting_frame << " ";
      data_stream << last_frame << " ";
    }

  // It will be good to assert that there will be no writing error.
  for (unsigned int frame_num = starting_frame; frame_num <= last_frame; ++frame_num)
    {
      data_stream << "\n";
      for (int param_num = 1; param_num <= num_conv_params; ++param_num)
        data_stream << this->_model_array[param_num][frame_num] << " ";
    }
  data_stream.close();
  return Succeeded::yes;
}

template <int num_conv_points>
void
GeneralizedPatlakMatrix<num_conv_points>::set_model_array(const Array<2, float>& model_array)
{
  this->_model_array = model_array;
}

template <int num_conv_points>
Array<2, float>
GeneralizedPatlakMatrix<num_conv_points>::get_model_array() const
{
  return this->_model_array;
}

template <int num_conv_points>
void
GeneralizedPatlakMatrix<num_conv_points>::set_Hfunction_array(const Array<2, float>& Hfunction_array)
{
  this->_Hfunction_array = Hfunction_array;
}

template <int num_conv_points>
Array<2, float>
GeneralizedPatlakMatrix<num_conv_points>::get_Hfunction_array() const
{
  return this->_Hfunction_array;
}

template <int num_conv_points>
void
GeneralizedPatlakMatrix<num_conv_points>::set_Ki_array(const Array<2, float>& Ki_array)
{
  this->_Ki_array = Ki_array;
}

template <int num_conv_points>
Array<2, float>
GeneralizedPatlakMatrix<num_conv_points>::get_Ki_array() const
{
  return this->_Ki_array;
}

template <int num_conv_points>
void
GeneralizedPatlakMatrix<num_conv_points>::set_conv_sample_interval(const unsigned int conv_sampling_interval)
{
  this->_conv_sampling_interval = conv_sampling_interval;
}

template <int num_conv_points>
void
GeneralizedPatlakMatrix<num_conv_points>::set_prefetched_sampling(const float kloss_start,
                                                                  const float kloss_end,
                                                                  const float kloss_nsamples)
{
  this->_kloss_start = kloss_start;
  this->_kloss_end = kloss_end;
  this->_kloss_nsamples = kloss_nsamples;
}

template <int num_conv_points>
const VectorWithOffset<float>
GeneralizedPatlakMatrix<num_conv_points>::get_model_array_sum() const
{
  BasicCoordinate<2, int> model_array_min, model_array_max;
  if (!(this->_model_array).get_regular_range(model_array_min, model_array_max))
    error("Model array has not regular range");
  VectorWithOffset<float> sum(model_array_min[1], model_array_max[1]);
  for (int param_num = model_array_min[1]; param_num <= model_array_max[1]; ++param_num)
    {
      sum[param_num] = 0.F;
      for (int frame_num = model_array_min[2]; frame_num <= model_array_max[2]; ++frame_num)
        sum[param_num] += this->_model_array[param_num][frame_num];
    }
  return sum;
}

template <int num_conv_points>
void
GeneralizedPatlakMatrix<num_conv_points>::threshold_model_array(const float threshold_value)
{
  BasicCoordinate<2, int> model_array_min, model_array_max;
  if (!(this->_model_array).get_regular_range(model_array_min, model_array_max))
    error("Model array has not regular range");

  for (int param_num = model_array_min[1]; param_num <= model_array_max[1]; ++param_num)
    for (int frame_num = model_array_min[2]; frame_num <= model_array_max[2]; ++frame_num)
      if (this->_model_array[param_num][frame_num] <= 0)
        this->_model_array[param_num][frame_num] = threshold_value;
}

template <int num_conv_points>
void
GeneralizedPatlakMatrix<num_conv_points>::set_if_uncalibrated(const bool is_uncalibrated)
{
  this->_is_uncalibrated = is_uncalibrated;
}

template <int num_conv_points>
void
GeneralizedPatlakMatrix<num_conv_points>::set_if_in_correct_scale(const bool in_correct_scale)
{
  this->_in_correct_scale = in_correct_scale;
}

template <int num_conv_points>
void
GeneralizedPatlakMatrix<num_conv_points>::set_matrix_in_total_frame_counts(const bool is_converted_to_total_counts)
{
  this->_is_converted_to_total_counts = is_converted_to_total_counts;
}

template <int num_conv_points>
void
GeneralizedPatlakMatrix<num_conv_points>::uncalibrate(const float cal_factor)
{
  if (this->_is_uncalibrated)
    warning("GeneralizedPatlakMatrix is already uncalibrated, so it will be not re-uncalibrated.");
  else
    {
      BasicCoordinate<2, int> model_array_min, model_array_max;
      if (!(this->_model_array).get_regular_range(model_array_min, model_array_max))
        error("Model array has not regular range");

      for (int param_num = model_array_min[1]; param_num <= model_array_max[1]; ++param_num)
        for (int frame_num = model_array_min[2]; frame_num <= model_array_max[2]; ++frame_num)
          this->_model_array[param_num][frame_num] /= cal_factor;

      GeneralizedPatlakMatrix::set_if_uncalibrated(true);
    }
}

template <int num_conv_points>
void
GeneralizedPatlakMatrix<num_conv_points>::scale_model_matrix(const float scale_factor)
{
  if (this->_in_correct_scale)
    warning("GeneralizedPatlakMatrix is already scaled, so it will not be re-scaled. ");
  else
    {
      BasicCoordinate<2, int> model_array_min, model_array_max;
      if (!(this->_model_array).get_regular_range(model_array_min, model_array_max))
        error("Model array has not regular range");
      for (int param_num = model_array_min[1]; param_num <= model_array_max[1]; ++param_num)
        for (int frame_num = model_array_min[2]; frame_num <= model_array_max[2]; ++frame_num)
          this->_model_array[param_num][frame_num] *= scale_factor;

      this->_in_correct_scale = true;
    }
}

template <int num_conv_points>
void
GeneralizedPatlakMatrix<num_conv_points>::convert_to_total_frame_counts(const TimeFrameDefinitions& time_frame_definitions)
{
  if (this->_is_converted_to_total_counts == true)
    warning("GeneralizedPatlakMatrix is already converted to total counts, so it will not be re-converted. ");
  else
    {
      BasicCoordinate<2, int> model_array_min, model_array_max;
      if (!(this->_model_array).get_regular_range(model_array_min, model_array_max))
        error("Model array has not regular range");
      for (int param_num = model_array_min[1]; param_num <= model_array_max[1]; ++param_num)
        for (int frame_num = model_array_min[2]; frame_num <= model_array_max[2]; ++frame_num)
          this->_model_array[param_num][frame_num] *= static_cast<float>(time_frame_definitions.get_duration(frame_num));

      this->_is_converted_to_total_counts = true;
    }
}

template <int num_conv_points>
void
GeneralizedPatlakMatrix<num_conv_points>::set_time_vector(const VectorWithOffset<float>& time_vector)
{
  this->_time_vector = time_vector;
}

template <int num_conv_points>
VectorWithOffset<float>
GeneralizedPatlakMatrix<num_conv_points>::get_time_vector() const
{
  return this->_time_vector;
}

template <int num_conv_points>
void
GeneralizedPatlakMatrix<num_conv_points>::estimate_inverse_Hfunction(float& kloss_estimate, float& Hfunction_val) const
{
  // Below we utilize the fact that Hfunction is monotonically decreasing
  for (unsigned int kloss_index = 1; kloss_index <= this->_kloss_nsamples; ++kloss_index)
    {
      // If the second column of this->_Hfunction_array is already in log scale (which is the default case),
      // then the Hfunction_val must also be converted to log scale, to conduct linear-log interpolation below
      // Hfunction_val=log(Hfunction_val);

      if (Hfunction_val < this->_Hfunction_array[2][kloss_index])
        {
          if (kloss_index < this->_kloss_nsamples)
            continue;
          else
            {
              kloss_estimate = this->_Hfunction_array[1][kloss_index];
              // cerr << "\nWARNING: Estimated H value (" << Hfunction_val << ") caused out-of-bounds kloss estimation: \n"
              //      << "[ub,lb]: [" << this->_kloss_start << "," << this->_kloss_end << "] , while est. kloss value=" <<
              //      kloss_estimate << "\n\n";
              break;
            }
        }
      else if (Hfunction_val > this->_Hfunction_array[2][kloss_index])
        {
          if (kloss_index > 1)
            {
              // Simple averaging
              // kloss_estimate=0.5*(this->_Hfunction_array[1][kloss_index-1]+this->_Hfunction_array[1][kloss_index]);

              // Linear interpolation, if second column of this->_Hfunction_array and Hfunction_val are in linear scale (not
              // implemented now)
              // Effectively Linear-log interpolation if second column of this->_Hfunction_array and Hfunction_val are in log
              // scale	(deafult now)
              kloss_estimate = this->_Hfunction_array[1][kloss_index - 1]
                               + ((this->_Hfunction_array[1][kloss_index] - this->_Hfunction_array[1][kloss_index - 1])
                                  / (this->_Hfunction_array[2][kloss_index] - this->_Hfunction_array[2][kloss_index - 1]))
                                     * (Hfunction_val - this->_Hfunction_array[2][kloss_index - 1]);
            }
          else
            {
              kloss_estimate = this->_Hfunction_array[1][kloss_index];
              // cerr << "\nWARNING: Estimated H value (" << Hfunction_val << ") caused out-of-bounds kloss estimation: \n"
              //      << "[ub,lb]: [" << this->_kloss_start << "," << this->_kloss_end << "] , while est. kloss value=" <<
              //      kloss_estimate << "\n\n";
            }
          break;
        }
      else
        {
          kloss_estimate = this->_Hfunction_array[1][kloss_index];
          break;
        }

      // cerr << "estimate_inverse_Hfunction: estimated kloss = " << kloss_estimate << endl;
    }
}

template <int num_conv_points>
void
GeneralizedPatlakMatrix<num_conv_points>::estimate_denominator_Kifunction(float& Ki_denominator, float& kloss_val) const
{
  // Below we utilize the fact that denominator_Kifunction is monotonically decreasing
  for (unsigned int kloss_index = 1; kloss_index <= this->_kloss_nsamples; ++kloss_index)
    {
      if (kloss_val == this->_Ki_array[1][kloss_index])
        {
          Ki_denominator = this->_Ki_array[2][kloss_index];
          break;
        }
      else if (kloss_val > this->_Ki_array[1][kloss_index])
        {
          if (kloss_index < this->_kloss_nsamples)
            continue;
          else
            {
              Ki_denominator = this->_Ki_array[2][kloss_index];
              // cerr << "\nWARNING: Estimated Ki denominator value (" << Ki_denominator << ") caused out-of-bounds kloss
              // estimation: \n"
              //      << "[ub,lb]: [" << this->_kloss_start << "," << this->_kloss_end << "] , while est. kloss value=" <<
              //      this->_Ki_array[1][kloss_index] << "\n\n";
              break;
            }
        }
      else if (kloss_val < this->_Ki_array[1][kloss_index])
        {
          if (kloss_index > 1)
            {
              // Simple averaging
              // Ki_denominator=0.5*(this->_Ki_array[2][kloss_index-1]+this->_Ki_array[2][kloss_index]);

              // Linear interpolation
              Ki_denominator = this->_Ki_array[2][kloss_index - 1]
                               + ((this->_Ki_array[2][kloss_index] - this->_Ki_array[2][kloss_index - 1])
                                  / (this->_Ki_array[1][kloss_index] - this->_Ki_array[1][kloss_index - 1]))
                                     * (kloss_val - this->_Ki_array[1][kloss_index - 1]);

              // Fast linear-log interpolation assuming column: this->_Ki_array[2][kloss_index] is in log scale already
              // float log_Ki_denominator=this->_Ki_array[2][kloss_index-1] + ((this->_Ki_array[2][kloss_index] -
              // this->_Ki_array[2][kloss_index-1])/(this->_Ki_array[1][kloss_index] -
              // this->_Ki_array[1][kloss_index-1]))*(kloss_val - this->_Ki_array[1][kloss_index-1]);
              // Ki_denominator=exp(log_Ki_denominator);

              // Linear-Log Interpolation
              // float log_Ki_denominator=log(this->_Ki_array[2][kloss_index-1]) + ((log(this->_Ki_array[2][kloss_index]) -
              // log(this->_Ki_array[2][kloss_index-1]))/(this->_Ki_array[1][kloss_index] -
              // this->_Ki_array[1][kloss_index-1]))*(kloss_val - this->_Ki_array[1][kloss_index-1]);
              // Ki_denominator=exp(log_Ki_denominator);
            }
          else
            {
              Ki_denominator = this->_Ki_array[2][kloss_index];
              // cerr << "\nWARNING: Estimated Ki denominator value (" << Ki_denominator << ") resulted from out-of-bounds kloss
              // estimation: \n"
              //      << "[ub,lb]: [" << this->_kloss_start << "," << this->_kloss_end << "] , while est. kloss value=" <<
              //      this->_Ki_array[1][kloss_index] << "\n\n";
            }
          break;
        }
    }

  // cerr << "estimate_denominator_Kifunction: estimated Ki_denominator = " << Ki_denominator << endl;
}

template <int num_conv_points>
void
GeneralizedPatlakMatrix<num_conv_points>::multiply_dynamic_image_with_model_and_add_to_input(
    DynamicDiscretisedDensity& impulse_response_image, const DynamicDiscretisedDensity& dynamic_image, int num_conv_params) const
{
  BasicCoordinate<2, int> model_array_min, model_array_max;
  if (!this->_model_array.get_regular_range(model_array_min, model_array_max))
    error("Model array has not regular range");

  // Assert that the sizes of the one frame of the dynamic image is equal with the parametric image size.
  // ChT::ToDo::Might be better to assert that each of the dimensions sizes with their voxle sizes are equal.
  // Could probably use has_same_characteristics()?
  assert(dynamic_image[1].size_all() == parametric_image.size_all());
  assert(dynamic_image.get_time_frame_definitions().get_num_frames() == static_cast<unsigned int>(model_array_max[2]));
  assert(model_array_max[1] - model_array_min[1] + 1 == num_conv_params);

  const int min_k_index = dynamic_image[1].get_min_index();
  const int max_k_index = dynamic_image[1].get_max_index();
  for (int k = min_k_index; k <= max_k_index; ++k)
    {
      const int min_j_index = dynamic_image[1][k].get_min_index();
      const int max_j_index = dynamic_image[1][k].get_max_index();
      for (int j = min_j_index; j <= max_j_index; ++j)
        {
          const int min_i_index = dynamic_image[1][k][j].get_min_index();
          const int max_i_index = dynamic_image[1][k][j].get_max_index();
          for (int i = min_i_index; i <= max_i_index; ++i)
            {
              // Estimation of impulse response image (the last slice is the estimated V parameter)
              for (int conv_param_num = model_array_min[1]; conv_param_num <= model_array_max[1]; ++conv_param_num)
                for (int frame_num = model_array_min[2]; frame_num <= model_array_max[2]; ++frame_num)
                  impulse_response_image[conv_param_num][k][j][i]
                      += this->_model_array[conv_param_num][frame_num] * dynamic_image[frame_num][k][j][i];
            }
        }
    }
}

template <int num_conv_points>
void
GeneralizedPatlakMatrix<num_conv_points>::multiply_dynamic_image_with_model(DynamicDiscretisedDensity& impulse_response_image,
                                                                            const DynamicDiscretisedDensity& dynamic_image,
                                                                            int num_conv_params) const
{
  std::fill(impulse_response_image.begin_all(), impulse_response_image.end_all(), 0.F);
  this->multiply_dynamic_image_with_model_and_add_to_input(impulse_response_image, dynamic_image, num_conv_params);
}

template <int num_conv_points>
void
GeneralizedPatlakMatrix<num_conv_points>::synthesize_impulse_response_from_parametric_image(
    DynamicDiscretisedDensity& impulse_response_image,
    const GeneralizedPatlakVoxelsOnCartesianGrid& parametric_image,
    int num_conv_params) const
{
  BasicCoordinate<2, int> model_array_min, model_array_max;
  if (!(this->_model_array).get_regular_range(model_array_min, model_array_max))
    error("Model array does not have a regular range");

  assert(model_array_max[1] - model_array_min[1] + 1 == num_conv_params);

  const int min_k_index = parametric_image.construct_single_density(1).get_min_index();
  const int max_k_index = parametric_image.construct_single_density(1).get_max_index();
  for (int k = min_k_index; k <= max_k_index; ++k)
    {
      const int min_j_index = (parametric_image.construct_single_density(1))[k].get_min_index();
      const int max_j_index = (parametric_image.construct_single_density(1))[k].get_max_index();
      for (int j = min_j_index; j <= max_j_index; ++j)
        {
          const int min_i_index = (parametric_image.construct_single_density(1))[k][j].get_min_index();
          const int max_i_index = (parametric_image.construct_single_density(1))[k][j].get_max_index();
          for (int i = min_i_index; i <= max_i_index; ++i)
            {
              for (int conv_param_num = model_array_min[1]; conv_param_num <= model_array_max[1] - 1; ++conv_param_num)
                {
                  int actual_time_point = (conv_param_num - 1) * this->_conv_sampling_interval + 1;
                  impulse_response_image[conv_param_num][k][j][i]
                      = parametric_image[k][j][i][1] * exp(-parametric_image[k][j][i][2] * actual_time_point);
                }
              impulse_response_image[num_conv_params][k][j][i] = parametric_image[k][j][i][3];
            }
        }
    }
}

template <int num_conv_points>
void
GeneralizedPatlakMatrix<num_conv_points>::multiply_impulse_response_with_model_and_add_to_input(
    DynamicDiscretisedDensity& dynamic_image, const DynamicDiscretisedDensity& impulse_response_image, int num_conv_params) const
{
  BasicCoordinate<2, int> model_array_min, model_array_max;
  if (!(this->_model_array).get_regular_range(model_array_min, model_array_max))
    error("Model array does not have a regular range");

  // Assert that the sizes of the one frame of the dynamic image is equal with the parametric image size.
  // ChT::ToDo::Might be better to assert that each of the dimensions sizes with their voxle sizes are equal.
  // Maybe this will be easier if I clone the single images for the two and then compare them.
  assert(dynamic_image.get_time_frame_definitions().get_num_frames() == static_cast<unsigned int>(model_array_max[2]));
  assert(model_array_max[1] - model_array_min[1] + 1 == num_conv_params);

  const int min_k_index = dynamic_image[1].get_min_index();
  const int max_k_index = dynamic_image[1].get_max_index();
  for (int k = min_k_index; k <= max_k_index; ++k)
    {
      const int min_j_index = dynamic_image[1][k].get_min_index();
      const int max_j_index = dynamic_image[1][k].get_max_index();
      for (int j = min_j_index; j <= max_j_index; ++j)
        {
          const int min_i_index = dynamic_image[1][k][j].get_min_index();
          const int max_i_index = dynamic_image[1][k][j].get_max_index();
          for (int i = min_i_index; i <= max_i_index; ++i)
            for (int frame_num = model_array_min[2]; frame_num <= model_array_max[2]; ++frame_num)
              for (int conv_param_num = model_array_min[1]; conv_param_num <= model_array_max[1]; ++conv_param_num)
                dynamic_image[frame_num][k][j][i]
                    += this->_model_array[conv_param_num][frame_num] * impulse_response_image[conv_param_num][k][j][i];
        }
    }
}

template <int num_conv_points>
void
GeneralizedPatlakMatrix<num_conv_points>::multiply_impulse_response_with_model(
    DynamicDiscretisedDensity& dynamic_image, const DynamicDiscretisedDensity& impulse_response_image, int num_conv_params) const
{
  std::fill(dynamic_image.begin_all(), dynamic_image.end_all(), 0.F);
  this->multiply_impulse_response_with_model_and_add_to_input(dynamic_image, impulse_response_image, num_conv_params);
}

template <int num_conv_points>
void
GeneralizedPatlakMatrix<num_conv_points>::multiply_parametric_image_with_model_and_add_to_input(
    DynamicDiscretisedDensity& dynamic_image,
    const GeneralizedPatlakVoxelsOnCartesianGrid& parametric_image,
    int num_conv_params) const
{
  BasicCoordinate<2, int> model_array_min, model_array_max;
  if (!(this->_model_array).get_regular_range(model_array_min, model_array_max))
    error("Model array does not have a regular range");

  VectorWithOffset<float> impulse_response_vector(model_array_min[1], model_array_max[1]);
  // Assert that the sizes of the one frame of the dynamic image is equal with the parametric image size.
  // ChT::ToDo::Might be better to assert that each of the dimensions sizes with their voxle sizes are equal.
  // Maybe this will be easier if I clone the single images for the two and then compare them.
  assert(dynamic_image[1].size_all() == parametric_image.size_all());
  assert(dynamic_image.get_time_frame_definitions().get_num_frames() == static_cast<unsigned int>(model_array_max[2]));
  assert(model_array_max[1] - model_array_min[1] + 1 == num_conv_params);

  const int min_k_index = dynamic_image[1].get_min_index();
  const int max_k_index = dynamic_image[1].get_max_index();
  for (int k = min_k_index; k <= max_k_index; ++k)
    {
      const int min_j_index = dynamic_image[1][k].get_min_index();
      const int max_j_index = dynamic_image[1][k].get_max_index();
      for (int j = min_j_index; j <= max_j_index; ++j)
        {
          const int min_i_index = dynamic_image[1][k][j].get_min_index();
          const int max_i_index = dynamic_image[1][k][j].get_max_index();
          for (int i = min_i_index; i <= max_i_index; ++i)
            {
              for (int conv_param_num = model_array_min[1]; conv_param_num <= model_array_max[1] - 1; ++conv_param_num)
                {
                  int actual_time_point = (conv_param_num - 1) * this->_conv_sampling_interval + 1;
                  impulse_response_vector[conv_param_num]
                      = parametric_image[k][j][i][1] * exp(-parametric_image[k][j][i][2] * actual_time_point);
                }
              impulse_response_vector[num_conv_params] = parametric_image[k][j][i][3];
              for (int frame_num = model_array_min[2]; frame_num <= model_array_max[2]; ++frame_num)
                for (int conv_param_num = model_array_min[1]; conv_param_num <= model_array_max[1]; ++conv_param_num)
                  dynamic_image[frame_num][k][j][i]
                      += this->_model_array[conv_param_num][frame_num] * impulse_response_vector[conv_param_num];
            }
        }
    }

  // Print out the min and max values of the last voxel impulse response vector
  const float current_min_imp_response = *std::min_element(impulse_response_vector.begin(), impulse_response_vector.end());
  const float current_max_imp_response = *std::max_element(impulse_response_vector.begin(), impulse_response_vector.end());
  cerr << "Impulse response vector initialized from parametric image: "
       << "(min, max): (" << current_min_imp_response << ", " << current_max_imp_response << ")" << endl;
}

template <int num_conv_points>
void
GeneralizedPatlakMatrix<num_conv_points>::multiply_parametric_image_with_model(
    DynamicDiscretisedDensity& dynamic_image,
    const GeneralizedPatlakVoxelsOnCartesianGrid& parametric_image,
    int num_conv_params) const
{
  std::fill(dynamic_image.begin_all(), dynamic_image.end_all(), 0.F);
  this->multiply_parametric_image_with_model_and_add_to_input(dynamic_image, parametric_image, num_conv_params);
}

template <int num_conv_points>
void
GeneralizedPatlakMatrix<num_conv_points>::estimate_generalized_patlak_parameters_with_impulse_response_and_add_to_input(
    GeneralizedPatlakVoxelsOnCartesianGrid& parametric_image,
    const DynamicDiscretisedDensity& impulse_response_image,
    int num_conv_params) const
{
  // Initialization
  BasicCoordinate<2, int> model_array_min, model_array_max;
  if (!(this->_model_array).get_regular_range(model_array_min, model_array_max))
    error("Model array does not have a regular range");

  const int min_k_index = parametric_image.construct_single_density(1).get_min_index();
  const int max_k_index = parametric_image.construct_single_density(1).get_max_index();
  assert(model_array_max[1] - model_array_min[1] + 1 == num_conv_params);
  for (int k = min_k_index; k <= max_k_index; ++k)
    {
      const int min_j_index = (parametric_image.construct_single_density(1))[k].get_min_index();
      const int max_j_index = (parametric_image.construct_single_density(1))[k].get_max_index();
      for (int j = min_j_index; j <= max_j_index; ++j)
        {
          const int min_i_index = (parametric_image.construct_single_density(1))[k][j].get_min_index();
          const int max_i_index = (parametric_image.construct_single_density(1))[k][j].get_max_index();
          for (int i = min_i_index; i <= max_i_index; ++i)
            {
              float SUM1 = 0.F, SUM2 = 0.F, Hfunction = 0.F, kloss_estimate = 0.F, Ki_denominator = 0.F;
              // Estimation of kloss parameter
              for (int conv_param_num = model_array_min[1]; conv_param_num <= model_array_max[1] - 1; ++conv_param_num)
                {
                  int actual_time_point = (conv_param_num - 1) * this->_conv_sampling_interval + 1;
                  SUM1 += actual_time_point * impulse_response_image[conv_param_num][k][j][i];
                  SUM2 += impulse_response_image[conv_param_num][k][j][i];
                }
              Hfunction = SUM1 / SUM2;
              this->estimate_inverse_Hfunction(kloss_estimate, Hfunction);
              parametric_image[k][j][i][2] += kloss_estimate;

              // Estimation of Ki parameter
              // Either utilize a look-up table
              // this->estimate_denominator_Kifunction(Ki_denominator,kloss_estimate);

              // or estimate the Ki denominator sum at each voxel
              for (int conv_param_num = model_array_min[1]; conv_param_num <= model_array_max[1] - 1; ++conv_param_num)
                {
                  int actual_time_point = (conv_param_num - 1) * this->_conv_sampling_interval + 1;
                  Ki_denominator += exp(-kloss_estimate * actual_time_point);
                }
              parametric_image[k][j][i][1] += SUM2 / Ki_denominator;

              // Estimation of V parameter
              int conv_param_num = model_array_max[1];
              parametric_image[k][j][i][3] += impulse_response_image[conv_param_num][k][j][i];
            }
        }
    }
}

template <int num_conv_points>
void
GeneralizedPatlakMatrix<num_conv_points>::estimate_generalized_patlak_parameters_with_impulse_response(
    GeneralizedPatlakVoxelsOnCartesianGrid& parametric_image,
    const DynamicDiscretisedDensity& impulse_response_image,
    int num_conv_params) const
{
  std::fill(parametric_image.begin_all(), parametric_image.end_all(), 0.F);
  this->estimate_generalized_patlak_parameters_with_impulse_response_and_add_to_input(
      parametric_image, impulse_response_image, num_conv_params);
}

template <int num_conv_points>
void
GeneralizedPatlakMatrix<num_conv_points>::normalise_parametric_image_with_model_sum(
    GeneralizedPatlakVoxelsOnCartesianGrid& parametric_image_out,
    const GeneralizedPatlakVoxelsOnCartesianGrid& parametric_image,
    int num_conv_params) const
{
  BasicCoordinate<2, int> model_array_min, model_array_max;
  if (!(this->_model_array).get_regular_range(model_array_min, model_array_max))
    error("Model array has not regular range");

  assert(parametric_image_out.size_all() == parametric_image.size_all());
  assert(model_array_max[1] - model_array_min[1] + 1 == num_conv_params);

  const int min_k_index = parametric_image.construct_single_density(1).get_min_index();
  const int max_k_index = parametric_image.construct_single_density(1).get_max_index();
  for (int k = min_k_index; k <= max_k_index; ++k)
    {
      const int min_j_index = (parametric_image.construct_single_density(1))[k].get_min_index();
      const int max_j_index = (parametric_image.construct_single_density(1))[k].get_max_index();
      for (int j = min_j_index; j <= max_j_index; ++j)
        {
          const int min_i_index = (parametric_image.construct_single_density(1))[k][j].get_min_index();
          const int max_i_index = (parametric_image.construct_single_density(1))[k][j].get_max_index();
          for (int i = min_i_index; i <= max_i_index; ++i)
            {
              parametric_image_out[k][j][i][1] = parametric_image[k][j][i][1] / ((this->get_model_array_sum())[1]);
              parametric_image_out[k][j][i][2] = parametric_image[k][j][i][2] / ((this->get_model_array_sum())[2]);
            }
        }
    }
}

template <int num_conv_points>
void
GeneralizedPatlakMatrix<num_conv_points>::estimate_nested_loop_parameters_with_model(
    GeneralizedPatlakVoxelsOnCartesianGrid& parametric_image,
    DynamicDiscretisedDensity& dynamic_image_nested_loop_estimate,
    DynamicDiscretisedDensity& dynamic_image_update_factor,
    const DynamicDiscretisedDensity& dynamic_image_reference,
    int num_nested_subiterations,
    float min_nested_rel_change,
    float max_nested_rel_change,
    int num_conv_params) const
{
  // Initialization
  BasicCoordinate<2, int> model_array_min, model_array_max;
  if (!(this->_model_array).get_regular_range(model_array_min, model_array_max))
    error("Model array does not have a regular range");

  VectorWithOffset<float> impulse_response_vector(model_array_min[1], model_array_max[1]);
  VectorWithOffset<float> impulse_response_update_factor(model_array_min[1], model_array_max[1]);
  VectorWithOffset<float> impulse_response_estimate(model_array_min[1], model_array_max[1]);
  VectorWithOffset<float> model_sensitivity_vector(model_array_min[1], model_array_max[1]);

  // Assert that the sizes of the one frame of the dynamic image is equal with the parametric image size.
  // ChT::ToDo::Might be better to assert that each of the dimensions sizes with their voxle sizes are equal.
  // Maybe this will be easier if I clone the single images for the two and then compare them.
  assert(dynamic_image_nested_loop_estimate[1].size_all() == parametric_image.size_all());
  assert(dynamic_image_nested_loop_estimate.get_time_frame_definitions().get_num_frames()
         == static_cast<unsigned int>(model_array_max[2]));
  assert(model_array_max[1] - model_array_min[1] + 1 == num_conv_params);

  // nested EM loop
  cerr << endl << "Entering nested loop " << endl;
  for (int nested_subiterations_num = 1; nested_subiterations_num <= num_nested_subiterations; nested_subiterations_num++)
    {
      // Forward-projection to transfer from parametric space to impulse response space and then dynamic (time) space
      const int min_k_index = dynamic_image_nested_loop_estimate[1].get_min_index();
      const int max_k_index = dynamic_image_nested_loop_estimate[1].get_max_index();
      for (int k = min_k_index; k <= max_k_index; ++k)
        {
          const int min_j_index = dynamic_image_nested_loop_estimate[1][k].get_min_index();
          const int max_j_index = dynamic_image_nested_loop_estimate[1][k].get_max_index();
          for (int j = min_j_index; j <= max_j_index; ++j)
            {
              const int min_i_index = dynamic_image_nested_loop_estimate[1][k][j].get_min_index();
              const int max_i_index = dynamic_image_nested_loop_estimate[1][k][j].get_max_index();
              for (int i = min_i_index; i <= max_i_index; ++i)
                {
                  for (int conv_param_num = model_array_min[1]; conv_param_num <= model_array_max[1]; ++conv_param_num)
                    {
                      if (conv_param_num < model_array_max[1])
                        impulse_response_vector[conv_param_num]
                            = parametric_image[k][j][i][1] * exp(-parametric_image[k][j][i][2] * conv_param_num);
                      else
                        impulse_response_vector[conv_param_num] = parametric_image[k][j][i][3];
                    }
                  for (int frame_num = model_array_min[2]; frame_num <= model_array_max[2]; ++frame_num)
                    {
                      float sum_over_conv_param = 0.F;
                      for (int conv_param_num = model_array_min[1]; conv_param_num <= model_array_max[1]; ++conv_param_num)
                        sum_over_conv_param
                            += impulse_response_vector[conv_param_num] * this->_model_array[conv_param_num][frame_num];
                      dynamic_image_nested_loop_estimate[frame_num][k][j][i] += sum_over_conv_param;
                    }
                }
            }
        }

      // Use the outer loop dynamic image estimate as a reference and divide it by the nested loop dynamic image estimate
      // to get the dynamic image update factor
      dynamic_image_update_factor = dynamic_image_reference;
      // loop over single_frame, utilize model_matrix and use the outer loop dynamic image estimate as a reference
      for (int frame_num = model_array_min[2]; frame_num <= model_array_max[2]; ++frame_num)
        divide(dynamic_image_update_factor[frame_num].begin_all(),
               dynamic_image_update_factor[frame_num].end_all(),
               dynamic_image_nested_loop_estimate[frame_num].begin_all(),
               small_num);

      // Back-projection of the dynamic image update factor to get the impulse reponse update factor
      // Also calculate sensitivity of the GeneralizedPatlakMatrix
      for (int k = min_k_index; k <= max_k_index; ++k)
        {
          const int min_j_index = dynamic_image_update_factor[1][k].get_min_index();
          const int max_j_index = dynamic_image_update_factor[1][k].get_max_index();
          for (int j = min_j_index; j <= max_j_index; ++j)
            {
              const int min_i_index = dynamic_image_update_factor[1][k][j].get_min_index();
              const int max_i_index = dynamic_image_update_factor[1][k][j].get_max_index();
              for (int i = min_i_index; i <= max_i_index; ++i)
                {
                  for (int conv_param_num = model_array_min[1]; conv_param_num <= model_array_max[1]; ++conv_param_num)
                    {
                      float sum_over_frames = 0.F;
                      float sensitivity_sum_over_frames = 0.F;
                      for (int frame_num = model_array_min[2]; frame_num <= model_array_max[2]; ++frame_num)
                        {
                          sum_over_frames
                              += this->_model_array[conv_param_num][frame_num] * dynamic_image_update_factor[frame_num][k][j][i];

                          // Also here we calculate the sensitivity of the GeneralizedPatlakMatrix
                          // TODO: No need to repeat sensitivity calculation for its voxel. This should be done elsewhere once and
                          // then passed to the method
                          sensitivity_sum_over_frames += this->_model_array[conv_param_num][frame_num];
                        }
                      impulse_response_update_factor[conv_param_num] = sum_over_frames;
                      model_sensitivity_vector[conv_param_num] = sensitivity_sum_over_frames;
                    }
                }
            }
        }

      // Sensitivity division of impulse reponse update factor
      for (int conv_param_num = model_array_min[1]; conv_param_num <= model_array_max[1]; ++conv_param_num)
        divide(impulse_response_update_factor.begin(),
               impulse_response_update_factor.end(),
               model_sensitivity_vector.begin(),
               small_num);

      if (nested_subiterations_num != 1)
        {
          const float current_min_nested_gradient
              = *std::min_element(impulse_response_update_factor.begin(), impulse_response_update_factor.end());
          const float current_max_nested_gradient
              = *std::max_element(impulse_response_update_factor.begin(), impulse_response_update_factor.end());
          const float new_min_nested_gradient = static_cast<float>(min_nested_rel_change);
          const float new_max_nested_gradient = static_cast<float>(max_nested_rel_change);
          cerr << "Nested iteration: " << nested_subiterations_num << " sub-gradient(update image) old value (min, max): ("
               << current_min_nested_gradient << ", " << current_max_nested_gradient << "), new value (min, max) ("
               << max(current_min_nested_gradient, new_min_nested_gradient) << ", "
               << min(current_max_nested_gradient, new_max_nested_gradient) << ")" << endl;

          threshold_upper_lower(impulse_response_update_factor.begin(),
                                impulse_response_update_factor.end(),
                                new_min_nested_gradient,
                                new_max_nested_gradient);
        }

      // Update the nested estimates of impulse reponse function
      for (int conv_param_num = model_array_min[1]; conv_param_num <= model_array_max[1]; ++conv_param_num)
        impulse_response_estimate[conv_param_num] *= impulse_response_update_factor[conv_param_num];

      // Print out the min and max values of the nested updated impulse reponse vector for each nested iteration
      const float current_min_nested_updated_impulse_response
          = *std::min_element(impulse_response_estimate.begin(), impulse_response_estimate.end());
      const float current_max_nested_updated_impulse_response
          = *std::max_element(impulse_response_estimate.begin(), impulse_response_estimate.end());
      cerr << "Nested iteration: " << nested_subiterations_num << " Updated impulse reponse value (min, max) ("
           << current_min_nested_updated_impulse_response << ", " << current_max_nested_updated_impulse_response << ")" << endl
           << endl;

      // Current nested loop estimation of the kinetic parameters Ki, kloss and V, based on updated impulse reponse estimate
      for (int k = min_k_index; k <= max_k_index; ++k)
        {
          const int min_j_index = dynamic_image_nested_loop_estimate[1][k].get_min_index();
          const int max_j_index = dynamic_image_nested_loop_estimate[1][k].get_max_index();
          for (int j = min_j_index; j <= max_j_index; ++j)
            {
              const int min_i_index = dynamic_image_nested_loop_estimate[1][k][j].get_min_index();
              const int max_i_index = dynamic_image_nested_loop_estimate[1][k][j].get_max_index();
              for (int i = min_i_index; i <= max_i_index; ++i)
                {
                  float SUM1 = 0.F, SUM2 = 0.F, Hfunction = 0.F, kloss_estimate = 0.F, Ki_denominator = 0.F;
                  // Estimation of kloss parameter
                  for (int conv_param_num = model_array_min[1]; conv_param_num <= model_array_max[1] - 1; ++conv_param_num)
                    {
                      SUM1 += conv_param_num * impulse_response_estimate[conv_param_num];
                      SUM2 += impulse_response_estimate[conv_param_num];
                    }
                  Hfunction = SUM1 / SUM2;
                  estimate_inverse_Hfunction(kloss_estimate, Hfunction);
                  parametric_image[k][j][i][2] = kloss_estimate;

                  // Estimation of Ki parameter
                  this->estimate_denominator_Kifunction(Ki_denominator, kloss_estimate);
                  parametric_image[k][j][i][1] = SUM2 / Ki_denominator;

                  // Estimation of V parameter
                  int conv_param_num = model_array_max[1];
                  parametric_image[k][j][i][3] = impulse_response_estimate[conv_param_num];
                }
            }
        }

      // Print out the min and max values of the nested updated impulse reponse vector for each nested iteration
      const float current_min_nested_updated_image = *std::min_element(parametric_image.begin_all(), parametric_image.end_all());
      const float current_max_nested_updated_image = *std::max_element(parametric_image.begin_all(), parametric_image.end_all());
      cerr << "Nested iteration: " << nested_subiterations_num << " Updated parametric image value (min, max) ("
           << current_min_nested_updated_image << ", " << current_max_nested_updated_image << ")" << endl
           << endl;
    }
}

END_NAMESPACE_STIR
