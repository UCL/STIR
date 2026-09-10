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
  \brief Implementations of inline functions of class stir::PatlakPlot
  \author Charalampos Tsoumpas

  \sa GeneralizedPatlakPlot.h, GeneralizedModelMatrix.h and KineticModel.h
*/

#include "stir/modelling/GeneralizedPatlakPlot.h"
#include <math.h>
#include "stir/format.h"

using namespace std;

START_NAMESPACE_STIR

void
GeneralizedPatlakPlot::set_defaults()
{
  base_type::set_defaults();
  this->_conv_sample_interval = 60;
  this->_kloss_lb = 0.0000001;
  this->_kloss_ub = 2;
  this->_kloss_num_samples = 1000000;
  with_initialization_loops = true;
}

const char* const GeneralizedPatlakPlot::registered_name = "Generalized Patlak Plot";

//! default constructor
GeneralizedPatlakPlot::GeneralizedPatlakPlot()
{
  this->_matrix_is_stored = false;
  this->_initialization_matrix_is_stored = false;
  this->set_defaults();
}

GeneralizedPatlakPlot::~GeneralizedPatlakPlot() //!< default destructor
{}

//! Simply get model matrix if it has been already stored
GeneralizedPatlakMatrix<2>
GeneralizedPatlakPlot::get_model_matrix() const
{
  if (_matrix_is_stored == false)
    error("It seems that ModelMatrix has not been set, yet. ");

  return _model_matrix;
}

//! Simply get model matrix if it has been already stored
ModelMatrix<2>
GeneralizedPatlakPlot::get_initialization_model_matrix() const
{
  if (_initialization_matrix_is_stored == false)
    error("It seems that initialization ModelMatrix has not been set, yet. ");

  return _initialization_model_matrix;
}

//! Simply set model matrix
void
GeneralizedPatlakPlot::set_model_matrix(GeneralizedPatlakMatrix<2> model_matrix)
{
  this->_model_matrix = model_matrix;
  this->_matrix_is_stored = true;
}

//! Simply set initialization model matrix
void
GeneralizedPatlakPlot::set_initialization_model_matrix(ModelMatrix<2> initialization_model_matrix)
{
  this->_initialization_model_matrix = initialization_model_matrix;
  this->_initialization_matrix_is_stored = true;
}

//! Simply set Hfunction matrix
void
GeneralizedPatlakPlot::set_Hfunction_matrix(GeneralizedPatlakMatrix<2> Hfunction_matrix)
{
  this->_Hfunction_matrix = Hfunction_matrix;
}

//! Simply get Hfunction matrix
GeneralizedPatlakMatrix<2>
GeneralizedPatlakPlot::get_Hfunction_matrix() const
{
  return _Hfunction_matrix;
}

//! Simply set Ki matrix
void
GeneralizedPatlakPlot::set_Ki_matrix(GeneralizedPatlakMatrix<2> Ki_matrix)
{
  this->_Ki_matrix = Ki_matrix;
}

//! Simply get Ki matrix
GeneralizedPatlakMatrix<2>
GeneralizedPatlakPlot::get_Ki_matrix() const
{
  return _Ki_matrix;
}

//! Create generalized Patlak model matrix from plasma data (has to be in appropriate frames)
GeneralizedPatlakMatrix<2>
GeneralizedPatlakPlot::get_model_matrix(const PlasmaData& complete_plasma_data,
                                        const PlasmaData& plasma_frame_data,
                                        const TimeFrameDefinitions& time_frame_definitions,
                                        const unsigned int starting_frame)
{
  // assert(starting_frame > 0);

  // if (_matrix_is_stored == false)
  //   {
  //     this->_starting_frame = starting_frame;
  //     BasicCoordinate<2, int> min_range;
  //     BasicCoordinate<2, int> max_range;
  //     unsigned int num_frames = plasma_frame_data.size();
  //     float last_frame_time
  //         = floor(0.5 * (time_frame_definitions.get_end_time(num_frames) + time_frame_definitions.get_start_time(num_frames)));
  //     unsigned int last_frame_mid_time = (unsigned int)floor(last_frame_time + 0.5);
  //     unsigned int num_conv_params = (unsigned int)floor(((last_frame_mid_time - 1) / this->_conv_sample_interval)) + 2;

  //     min_range[1] = 1;
  //     min_range[2] = this->_starting_frame;
  //     max_range[1] = num_conv_params;
  //     max_range[2] = num_frames;
  //     IndexRange<2> data_range(min_range, max_range);
  //     Array<2, float> patlak_array(data_range);
  //     VectorWithOffset<float> time_vector(min_range[2], max_range[2]);
  //     VectorWithOffset<float> plasma_sample_dec_fact(min_range[1], max_range[1]);
  //     VectorWithOffset<float> dec_fact(min_range[2], max_range[2]);
  //     PlasmaData::const_iterator cur_iter = plasma_frame_data.begin() + this->_starting_frame - 1;
  //     PlasmaData::const_iterator complete_plasma_cur_iter;

  //     unsigned int frame_num, conv_sample, actual_time_point;

  //     if (plasma_frame_data.get_is_decay_corrected())
  //       warning("Uncorrecting previous decay correction, while putting the plasma_data into the model_matrix.");
  //     else if (!plasma_frame_data.get_is_decay_corrected())
  //       error("plasma_data have not been corrected during the process, which will create wrong results!!!");

  //     std::cout << "\nCreating Generalized Model Matrix (Here printed in its transverse format)\n\n"
  //               << "NOTE1: It contains as many columns as the number of later frames participating in parameter estimation\n"
  //               << "It contains as many rows as the convolution points of the input function\n"
  //               << "+ 1 last row consisting of the plasma counts for the corresponding later frame\n"
  //               << "NOTE2: Last element of each column is the plasma counts for the corresponding later frame\n\n"
  //               << "First Column: plasma samples for frame 1	...		Last Column: plasma samples for last frame\n";

  //     // Fillling of the Patlak array.
  //     // First conv_sample columns are filled with plasma samples for each sec,
  //     for (frame_num = this->_starting_frame; cur_iter != this->_plasma_frame_data.end(); ++frame_num, ++cur_iter)
  //       {
  //         float cur_frame_time = 0.5 * (this->_frame_defs.get_end_time(frame_num) +
  //         this->_frame_defs.get_start_time(frame_num)); unsigned int cur_frame_mid_time = (int)floor(cur_frame_time + 0.5);
  //         std::cout << "\nFrame Number: " << frame_num << " Current Frame Mid Time (float): " << cur_frame_time
  //                   << " Current Frame Mid Time (int): " << cur_frame_mid_time << "\n";
  //         complete_plasma_cur_iter = this->_complete_plasma_data.begin() + cur_frame_mid_time - 1;

  //         for (conv_sample = 1, actual_time_point = 1; actual_time_point <= last_frame_mid_time; ++conv_sample)
  //           {
  //             actual_time_point = (conv_sample - 1) * this->_conv_sample_interval + 1;
  //             if (actual_time_point <= cur_frame_mid_time)
  //               patlak_array[conv_sample][frame_num]
  //                   = complete_plasma_cur_iter->get_plasma_counts_in_kBq() * this->_conv_sample_interval;
  //             else
  //               patlak_array[conv_sample][frame_num] = 0;

  //             complete_plasma_cur_iter = complete_plasma_cur_iter - this->_conv_sample_interval;
  //           }

  //         // Last column is filled with the plasma activity of the later frames
  //         patlak_array[num_conv_params][frame_num] = cur_iter->get_plasma_counts_in_kBq();

  //         std::cout << "\n";
  //         // Un-correcting for decay, if data are decay corrected
  //         if (this->_plasma_frame_data.get_is_decay_corrected())
  //           {
  //             cerr << "Timing info (Decay correction factor info)" << endl;
  //             complete_plasma_cur_iter = this->_complete_plasma_data.begin() + cur_frame_mid_time - 1;
  //             for (conv_sample = 1, actual_time_point = 1; actual_time_point <= last_frame_mid_time; ++conv_sample)
  //               {
  //                 actual_time_point = (conv_sample - 1) * this->_conv_sample_interval + 1;
  //                 if (actual_time_point <= cur_frame_mid_time)
  //                   {
  //                     cerr << complete_plasma_cur_iter->get_time_in_s() << "  ";
  //                     plasma_sample_dec_fact[conv_sample] = static_cast<float>(decay_correction_factor(
  //                         this->_complete_plasma_data.get_isotope_halflife(), complete_plasma_cur_iter->get_time_in_s()));
  //                     patlak_array[conv_sample][frame_num] /= plasma_sample_dec_fact[conv_sample];
  //                   }
  //                 else
  //                   {
  //                     cerr << 0 << "  ";
  //                     plasma_sample_dec_fact[conv_sample] = 1;
  //                   }

  //                 cerr << " (" << plasma_sample_dec_fact[conv_sample] << ")		";

  //                 complete_plasma_cur_iter = complete_plasma_cur_iter - this->_conv_sample_interval;
  //               }

  //             dec_fact[frame_num] = static_cast<float>(
  //                 decay_correction_factor(this->_plasma_frame_data.get_isotope_halflife(),
  //                                         this->_plasma_frame_data.get_time_frame_definitions().get_start_time(frame_num),
  //                                         this->_plasma_frame_data.get_time_frame_definitions().get_end_time(frame_num)));

  //             patlak_array[num_conv_params][frame_num] /= dec_fact[frame_num];
  //             time_vector[frame_num] = static_cast<float>(
  //                 0.5 * (this->_frame_defs.get_end_time(frame_num) + this->_frame_defs.get_start_time(frame_num)));
  //           }
  //       }
  //     std::cout << endl << endl;
  //     // Print out the model matrix
  //     for (conv_sample = 1; conv_sample <= num_conv_params; ++conv_sample)
  //       {
  //         for (frame_num = this->_starting_frame; frame_num <= num_frames; ++frame_num)
  //           std::cout << patlak_array[conv_sample][frame_num] << "			";
  //         std::cout << "\n";
  //       }

  //     if (this->_plasma_frame_data.get_is_decay_corrected())
  //       {
  //         std::cout << "\n\nFrame Index	Start Time	End Time	Decay correction factor\n";
  //         cur_iter = this->_plasma_frame_data.begin() + this->_starting_frame - 1;
  //         for (frame_num = this->_starting_frame; cur_iter != this->_plasma_frame_data.end(); ++frame_num, ++cur_iter)
  //           {
  //             // Print out the frame indices, start and time points and the decay correction factors
  //             std::cout << frame_num << "		"
  //                       << _plasma_frame_data.get_time_frame_definitions().get_start_time(frame_num) << "		"
  //                       << _plasma_frame_data.get_time_frame_definitions().get_end_time(frame_num) << "		"
  //                       << dec_fact[frame_num] << "\n";
  //           }
  //       }

  //     assert(frame_num - 1 == plasma_frame_data.size());
  //     this->_model_matrix.set_model_array(patlak_array);
  //     this->_model_matrix.set_conv_sample_interval(this->_conv_sample_interval);
  //     this->_model_matrix.set_time_vector(time_vector);
  //     this->_model_matrix.set_if_in_correct_scale(this->_in_correct_scale);
  //     this->_model_matrix.threshold_model_array(.0000001F);
  //     this->_matrix_is_stored = true;
  //   }
  // return _model_matrix;
}

//! Create initialization (standard Patlak) model matrix from plasma data (has to be in appropriate frames: i.e.
//! plasma_frame_data)
ModelMatrix<2>
GeneralizedPatlakPlot::get_initialization_model_matrix(const PlasmaData& plasma_frame_data,
                                                       const TimeFrameDefinitions& time_frame_definitions,
                                                       const unsigned int starting_frame)
{
  // assert(starting_frame > 0);

  // if (_matrix_is_stored == false)
  //   {
  //     this->_starting_frame = starting_frame;
  //     BasicCoordinate<2, int> min_range;
  //     BasicCoordinate<2, int> max_range;
  //     min_range[1] = 1;
  //     min_range[2] = starting_frame;
  //     max_range[1] = 2;
  //     max_range[2] = plasma_frame_data.size();
  //     IndexRange<2> data_range(min_range, max_range);
  //     Array<2, float> patlak_array(data_range);
  //     VectorWithOffset<float> time_vector(min_range[2], max_range[2]);
  //     PlasmaData::const_iterator cur_iter = plasma_frame_data.begin();

  //     double sum_value = 0.;
  //     unsigned int sample_num;
  //     //      std::cerr << "\n" << cur_iter->get_plasma_counts_in_kBq() << " " << cur_iter->get_time_in_s() << "\n";
  //     //      std::cerr <<
  //     //      "\nFrame-PlasmaStart-TimeFrameFileStart-PlasmaDuration-TimeFrameFileDuration-PlasmaEnd-TimeFrameFileEnd\n" ;
  //     for (sample_num = 1; sample_num < starting_frame; ++sample_num, ++cur_iter)
  //       {
  //         sum_value
  //             += cur_iter->get_plasma_counts_in_kBq() *
  //             plasma_frame_data.get_time_frame_definitions().get_duration(sample_num);
  //       }

  //     assert(cur_iter == plasma_frame_data.begin() + starting_frame - 1);

  //     for (sample_num = starting_frame; cur_iter != plasma_frame_data.end(); ++sample_num, ++cur_iter)
  //       {
  //         double integral_step
  //             = cur_iter->get_plasma_counts_in_kBq() * plasma_frame_data.get_time_frame_definitions().get_duration(sample_num);
  //         // Calculation of the plasma integral only up to the mid time of the current plasma frame
  //         sum_value += 0.5 * integral_step;
  //         // Fillling of the Patlak array. First column is filled with plasma integral, second column with plasma activity
  //         patlak_array[1][sample_num] = static_cast<float>(sum_value);
  //         patlak_array[2][sample_num] = cur_iter->get_plasma_counts_in_kBq();
  //         if (plasma_frame_data.get_is_decay_corrected())
  //           {
  //             const float dec_fact = static_cast<float>(
  //                 decay_correction_factor(plasma_frame_data.get_isotope_halflife(),
  //                                         plasma_frame_data.get_time_frame_definitions().get_start_time(sample_num),
  //                                         plasma_frame_data.get_time_frame_definitions().get_end_time(sample_num)));
  //             patlak_array[1][sample_num] /= dec_fact;
  //             patlak_array[2][sample_num] /= dec_fact;
  //             time_vector[sample_num] = static_cast<float>(
  //                 0.5 * (time_frame_definitions.get_end_time(sample_num) + time_frame_definitions.get_start_time(sample_num)));
  //           }
  //         // Completion of integral calculation before moving to the next plasma frame
  //         sum_value += 0.5 * integral_step;
  //       }
  //     if (plasma_frame_data.get_is_decay_corrected())
  //       warning("Uncorrecting previous decay correction, while putting the plasma_frame_data into the model_matrix.");
  //     else if (!plasma_frame_data.get_is_decay_corrected())
  //       warning("plasma_frame_data have not been corrected during the process, which might create wrong results!!!");

  //     assert(sample_num - 1 == plasma_frame_data.size());
  //     this->_initialization_model_matrix.set_model_array(patlak_array);
  //     this->_initialization_model_matrix.set_time_vector(time_vector);
  //     this->_initialization_model_matrix.set_is_in_correct_scale(this->_in_correct_scale);
  //     this->_initialization_model_matrix.threshold_model_array(.0000001F);
  //     this->_initialization_matrix_is_stored = true;
  //   }
  // return _initialization_model_matrix;
}

//! Create prefeched H matrix from private members
void
GeneralizedPatlakPlot::create_Hfunction_matrix()
{
  BasicCoordinate<2, int> min_range;
  BasicCoordinate<2, int> max_range;
  min_range[1] = 1;
  min_range[2] = 1;
  max_range[1] = 2;
  max_range[2] = this->_kloss_num_samples;
  IndexRange<2> kloss_range(min_range, max_range);
  Array<2, float> Hfunction_array(kloss_range);
  unsigned int kloss_index, conv_sample, actual_time_point;
  float kloss_step = (this->_kloss_ub - this->_kloss_lb) / this->_kloss_num_samples;
  float kloss_val = this->_kloss_lb;

  std::cout << "\nPrecalculating H Matrix for the following range of " << this->_kloss_num_samples << " kloss values:\n"
            << "[ kloss_start : kloss_end : kloss_step ] -> [ " << this->_kloss_lb << " : " << this->_kloss_ub << " : "
            << kloss_step << " ]\n\n";

  for (kloss_index = 1; kloss_index <= this->_kloss_num_samples; ++kloss_index)
    {
      float numerator_sum = 0, denominator_sum = 0;
      for (conv_sample = 1, actual_time_point = 1; actual_time_point <= this->_last_frame_ref_time; ++conv_sample)
        {
          actual_time_point = (conv_sample - 1) * this->_conv_sample_interval + 1;
          numerator_sum += actual_time_point * exp(-kloss_val * actual_time_point);
          denominator_sum += exp(-kloss_val * actual_time_point);
        }
      Hfunction_array[1][kloss_index] = kloss_val;

      // For linear interpolation, the linear H value is sufficient
      Hfunction_array[2][kloss_index] = numerator_sum / denominator_sum;

      // For fast linear-log interpolation, better pre-calculate the log value for H
      // Hfunction_array[2][kloss_index]=log(numerator_sum/denominator_sum);

      kloss_val += kloss_step;
    }
  this->_model_matrix.set_Hfunction_array(Hfunction_array);
  this->_model_matrix.set_prefetched_sampling(this->_kloss_lb, this->_kloss_ub, this->_kloss_num_samples);
  std::cout << "Precalculation of H Matrix has been completed\n\n";
}

//! Create prefeched Ki matrix from private members
void
GeneralizedPatlakPlot::create_Ki_matrix()
{
  BasicCoordinate<2, int> min_range;
  BasicCoordinate<2, int> max_range;
  min_range[1] = 1;
  min_range[2] = 1;
  max_range[1] = 2;
  max_range[2] = this->_kloss_num_samples;
  IndexRange<2> kloss_range(min_range, max_range);
  Array<2, float> Ki_array(kloss_range);
  unsigned int kloss_index, conv_sample, actual_time_point;
  float kloss_step = (this->_kloss_ub - this->_kloss_lb) / this->_kloss_num_samples;
  float kloss_val = this->_kloss_lb;

  std::cout << "\nPrecalculating Ki Matrix for the following range of " << this->_kloss_num_samples << " kloss values:\n"
            << "[ kloss_start : kloss_end : kloss_step ] -> [ " << this->_kloss_lb << " : " << this->_kloss_ub << " : "
            << kloss_step << " ]\n\n";

  for (kloss_index = 1; kloss_index <= this->_kloss_num_samples; ++kloss_index)
    {
      float Ki_sum = 0;
      for (conv_sample = 1, actual_time_point = 1; actual_time_point <= this->_last_frame_ref_time; ++conv_sample)
        {
          actual_time_point = (conv_sample - 1) * this->_conv_sample_interval + 1;
          Ki_sum += exp(-kloss_val * actual_time_point);
        }

      Ki_array[1][kloss_index] = kloss_val;
      Ki_array[2][kloss_index] = Ki_sum;
      kloss_val += kloss_step;
    }
  this->_model_matrix.set_Ki_array(Ki_array);
  // this->_model_matrix.set_prefetched_sampling(this->_kloss_lb,this->_kloss_ub,this->_kloss_num_samples);

  std::cout << "\nPrecalculation of Ki Matrix has been completed\n\n";
}

//! Create model matrix from private members
void
GeneralizedPatlakPlot::create_model_matrix()
{
  if (_matrix_is_stored == false)
    {
      base_type::create_model_matrix();

      BasicCoordinate<2, int> min_range;
      BasicCoordinate<2, int> max_range;
      min_range[1] = 1;
      min_range[2] = this->get_starting_frame();
      max_range[1] = this->_num_conv_params;
      max_range[2] = this->get_plasma_data().size();

      IndexRange<2> data_range(min_range, max_range);
      Array<2, float> patlak_array(data_range);
      VectorWithOffset<float> time_vector(min_range[2], max_range[2]);
      VectorWithOffset<float> plasma_sample_dec_fact(min_range[1], max_range[1]);
      VectorWithOffset<float> dec_fact(min_range[2], max_range[2]);
      PlasmaData::const_iterator cur_iter = this->_plasma_frame_data.begin() + this->get_starting_frame() - 1;
      PlasmaData::const_iterator complete_plasma_cur_iter;

      unsigned int frame_num, conv_sample, actual_time_point;
      const bool integrate_to_midpoint = this->get_frame_reference_time() == 1;

      // std::cout<< "The total number of frames are: " << this->_num_frames << "\n"
      // << "The total number of complete plasma samples are: " << this->_plasma_frame_data.size() << "\n"
      // << "The last frame middle time is : " << this->_last_frame_mid_time << "\n"
      // << "The time shift in complete plasma samples is: " << this->_plasma_frame_data.get_time_shift() << "\n"
      // << "The total number of convolution points + 1(one) more column are: " << this->_num_conv_params << "\n"
      // << "First Column: plasma samples for frame 1	...		Last Column: plasma samples for last frame\n";

      // Fillling of the Patlak array.
      // First conv_sample columns are filled with plasma samples for each sec,
      for (frame_num = this->get_starting_frame(); cur_iter != this->_plasma_frame_data.end(); ++frame_num, ++cur_iter)
        {
          const double frame_start = this->_plasma_frame_data.get_time_frame_definitions().get_start_time(frame_num);
          const double frame_end = this->_plasma_frame_data.get_time_frame_definitions().get_end_time(frame_num);

          // instant at which this frame's model value is evaluated
          const double cur_frame_ref_time_f = integrate_to_midpoint ? 0.5 * (frame_start + frame_end) : frame_end;

          unsigned int cur_frame_ref_time = static_cast<unsigned int>(floor(cur_frame_ref_time_f + 0.5));
          info(format("Frame Number: {} Current Frame Mid Time (float): {} Current Frame Mid Time (int): {} ",
                      frame_num,
                      cur_frame_ref_time_f,
                      cur_frame_ref_time));

          complete_plasma_cur_iter = this->_plasma_frame_data.begin() + cur_frame_ref_time - 1;

          for (conv_sample = 1, actual_time_point = 1; actual_time_point <= this->_last_frame_ref_time; ++conv_sample)
            {
              actual_time_point = (conv_sample - 1) * this->_conv_sample_interval + 1;
              if (actual_time_point <= cur_frame_ref_time)
                patlak_array[conv_sample][frame_num]
                    = complete_plasma_cur_iter->get_plasma_counts_in_kBq() * this->_conv_sample_interval;
              else
                patlak_array[conv_sample][frame_num] = 0;

              complete_plasma_cur_iter = complete_plasma_cur_iter - this->_conv_sample_interval;
            }

          // Last column is filled with the plasma activity of the later frames
          patlak_array[this->_num_conv_params][frame_num] = cur_iter->get_plasma_counts_in_kBq();

          // Un-correcting for decay, if data are decay corrected
          if (this->_plasma_frame_data.get_is_decay_corrected())
            {
              complete_plasma_cur_iter = this->_plasma_frame_data.begin() + cur_frame_ref_time - 1;
              for (conv_sample = 1, actual_time_point = 1; actual_time_point <= this->_last_frame_ref_time; ++conv_sample)
                {
                  actual_time_point = (conv_sample - 1) * this->_conv_sample_interval + 1;
                  if (actual_time_point <= cur_frame_ref_time)
                    {
                      cerr << complete_plasma_cur_iter->get_time_in_s() << "  ";
                      plasma_sample_dec_fact[conv_sample] = static_cast<float>(decay_correction_factor(
                          this->_plasma_frame_data.get_isotope_halflife(), complete_plasma_cur_iter->get_time_in_s()));
                      patlak_array[conv_sample][frame_num] /= plasma_sample_dec_fact[conv_sample];
                    }
                  else
                    {
                      cerr << 0 << "  ";
                      plasma_sample_dec_fact[conv_sample] = 1;
                    }

                  cerr << " (" << plasma_sample_dec_fact[conv_sample] << ")		";

                  complete_plasma_cur_iter = complete_plasma_cur_iter - this->_conv_sample_interval;
                }

              dec_fact[frame_num] = static_cast<float>(
                  decay_correction_factor(this->_plasma_frame_data.get_isotope_halflife(),
                                          this->_plasma_frame_data.get_time_frame_definitions().get_start_time(frame_num),
                                          this->_plasma_frame_data.get_time_frame_definitions().get_end_time(frame_num)));

              patlak_array[this->_num_conv_params][frame_num] /= dec_fact[frame_num];
              time_vector[frame_num] = static_cast<float>(0.5
                                                          * (this->get_time_frame_definitions().get_end_time(frame_num)
                                                             + this->get_time_frame_definitions().get_start_time(frame_num)));
            }
        }

      // Print out the model matrix
      for (conv_sample = 1; conv_sample <= this->_num_conv_params; ++conv_sample)
        {
          for (frame_num = this->get_starting_frame(); frame_num <= this->_num_frames; ++frame_num)
            std::cout << patlak_array[conv_sample][frame_num] << "			";
          std::cout << "\n";
        }

      if (this->_plasma_frame_data.get_is_decay_corrected())
        {
          std::cout << "\n\nFrame Index	Start Time	End Time	Decay correction factor\n";
          cur_iter = this->_plasma_frame_data.begin() + this->get_starting_frame() - 1;
          for (frame_num = this->get_starting_frame(); cur_iter != this->_plasma_frame_data.end(); ++frame_num, ++cur_iter)
            {
              // Print out the frame indices, start and time points and the decay correction factors
              std::cout << frame_num << "		"
                        << _plasma_frame_data.get_time_frame_definitions().get_start_time(frame_num) << "		"
                        << _plasma_frame_data.get_time_frame_definitions().get_end_time(frame_num) << "		"
                        << dec_fact[frame_num] << "\n";
            }
        }

      assert(frame_num - 1 == this->_plasma_frame_data.size());
      this->_model_matrix.set_model_array(patlak_array);
      this->_model_matrix.set_conv_sample_interval(this->_conv_sample_interval);
      this->_model_matrix.set_time_vector(time_vector);
      // Uncalibrate the ModelMatrix instead of Calibrating all the Dynamic Images. This should make faster the computation.
      // Supposes the images are not calibrated.
      this->_model_matrix.uncalibrate(this->_cal_factor);
      this->_model_matrix.set_matrix_in_total_frame_counts(this->_plasma_in_total_cnt);
      if (this->_in_total_cnt)
        this->_model_matrix.convert_to_total_frame_counts(this->get_time_frame_definitions());
      this->_model_matrix.set_if_in_correct_scale(this->_in_correct_scale);
      this->_model_matrix.threshold_model_array(.000000001F);
      this->_matrix_is_stored = true;
    }
  else
    warning("ModelMatrix has been already created");
}

Succeeded
GeneralizedPatlakPlot::set_up()
{
  if (base_type::set_up() != Succeeded::yes)
    return Succeeded::no;

  info("Preparing to set up the Generalized Patlak Plot...");
  // std::cout << "Set up of Generalized Patlak Plot has been completed." << endl;
  this->create_model_matrix();

  if (with_initialization_loops)
    {
      info("Preparing to set up the initialization (standard) Patlak Plot...");
      linear_model = std::make_shared<PatlakPlot>(this->get_exam_info_sptr());

      // linear_model->set_starting_frame(this->_starting_frame);
      // linear_model->set_cal_factor(this->_cal_factor);
      // linear_model->set_time_frame_definitions(this->_frame_defs);
      // linear_model->set_in_total_cnt(this->_in_total_cnt);
      // linear_model->set_in_correct_scale(this->_in_correct_scale);
      // linear_model->set_frame_reference_time(this->get_frame_reference_time());
      // // reuse the plasma data we already sampled — do not re-read the blood file
      // linear_model->set_plasma_data(this->_plasma_frame_data.get_sample_data_in_frames(this->_frame_defs));

      linear_model->set_up();
      info("Set up of initialization (standard) Patlak Plot has been completed.");
    }

  std::cout << "Preparing to construct look up tables..." << endl;
  this->create_Hfunction_matrix();
  this->create_Ki_matrix();
  std::cout << "Look up tables construction has been completed." << endl;

  if ((this->_matrix_is_stored == true) && (this->_initialization_matrix_is_stored == true))
    {
      std::cout << "Set up of generalized and initialization (standard) Patlak Plot is successful." << endl;
      return Succeeded::yes;
    }
  else if (this->_matrix_is_stored == false)
    {
      std::cout << "Set up of Generalized Patlak Plot has failed." << endl;
      return Succeeded::no;
    }
  else
    {
      std::cout << "Set up of initialization (standard) Patlak Plot has failed." << endl;
      return Succeeded::no;
    }
}

void
GeneralizedPatlakPlot::multiply_dynamic_image_with_model_gradient(DynamicDiscretisedDensity& impulse_response,
                                                                  const DynamicDiscretisedDensity& dyn_image) const
{
  if (!this->_in_correct_scale)
    {
#ifndef NDEBUG
      this->_model_matrix.write_to_file("patlak_matrix_not_in_correct_scale.txt", this->_num_conv_params);
#endif // NDEBUG
      const DiscretisedDensityOnCartesianGrid<3, float>* image_cartesian_ptr
          = dynamic_cast<DiscretisedDensityOnCartesianGrid<3, float>*>(((dyn_image.get_densities())[0]).get());
      const BasicCoordinate<3, float> this_grid_spacing = image_cartesian_ptr->get_grid_spacing();
      this->_model_matrix.scale_model_matrix(this_grid_spacing[2] / dyn_image.get_scanner_default_bin_size());
#ifndef NDEBUG
      this->_model_matrix.write_to_file("patlak_matrix_in_correct_scale.txt", this->_num_conv_params);
#endif // NDEBUG
    }
  this->_model_matrix.multiply_dynamic_image_with_model(impulse_response, dyn_image, this->_num_conv_params);
}

void
GeneralizedPatlakPlot::multiply_dynamic_image_with_model_gradient_and_add_to_input(
    DynamicDiscretisedDensity& impulse_response, const DynamicDiscretisedDensity& dyn_image) const
{
  if (!this->_in_correct_scale)
    {
#ifndef NDEBUG
      this->_model_matrix.write_to_file("patlak_matrix_not_in_correct_scale.txt", this->_num_conv_params);
#endif // NDEBUG
      const DiscretisedDensityOnCartesianGrid<3, float>* image_cartesian_ptr
          = dynamic_cast<DiscretisedDensityOnCartesianGrid<3, float>*>(((dyn_image.get_densities())[0]).get());
      const BasicCoordinate<3, float> this_grid_spacing = image_cartesian_ptr->get_grid_spacing();
      this->_model_matrix.scale_model_matrix(this_grid_spacing[2] / dyn_image.get_scanner_default_bin_size());
#ifndef NDEBUG
      this->_model_matrix.write_to_file("patlak_matrix_in_correct_scale.txt", this->_num_conv_params);
#endif // NDEBUG
    }
  this->_model_matrix.multiply_dynamic_image_with_model_and_add_to_input(impulse_response, dyn_image, this->_num_conv_params);
}

// Should be a virtual function declared in the KineticModels or better to the LinearModels
void
GeneralizedPatlakPlot::get_impulse_response_from_parametric_image(DynamicDiscretisedDensity& impulse_response_image,
                                                                  const Parametric3VoxelsOnCartesianGrid& par_image) const
{

  this->_model_matrix.synthesize_impulse_response_from_parametric_image(
      impulse_response_image, par_image, this->_num_conv_params);
}

// Should be a virtual function declared in the KineticModels or better to the LinearModels
void
GeneralizedPatlakPlot::get_dynamic_image_from_impulse_response(DynamicDiscretisedDensity& dyn_image,
                                                               const DynamicDiscretisedDensity& impulse_response_image) const
{
  if (!this->_in_correct_scale)
    {
#ifndef NDEBUG
      this->_model_matrix.write_to_file("patlak_matrix_not_in_correct_scale.txt", this->_num_conv_params);
#endif // NDEBUG
      const DiscretisedDensityOnCartesianGrid<3, float>* image_cartesian_ptr
          = dynamic_cast<DiscretisedDensityOnCartesianGrid<3, float>*>(((dyn_image.get_densities())[0]).get());
      const BasicCoordinate<3, float> this_grid_spacing = image_cartesian_ptr->get_grid_spacing();
      this->_model_matrix.scale_model_matrix(this_grid_spacing[2] / dyn_image.get_scanner_default_bin_size());
#ifndef NDEBUG
      this->_model_matrix.write_to_file("patlak_matrix_in_correct_scale.txt", this->_num_conv_params);
#endif // NDEBUG
    }

  this->_model_matrix.multiply_impulse_response_with_model(dyn_image, impulse_response_image, this->_num_conv_params);
}

// Should be a virtual function declared in the KineticModels or better to the LinearModels
void
GeneralizedPatlakPlot::get_dynamic_image_from_parametric_image(DynamicDiscretisedDensity& dyn_image,
                                                               const Parametric3VoxelsOnCartesianGrid& par_image) const
{
  if (!this->_in_correct_scale)
    {
#ifndef NDEBUG
      this->_model_matrix.write_to_file("patlak_matrix_not_in_correct_scale.txt", this->_num_conv_params);
#endif // NDEBUG
      const DiscretisedDensityOnCartesianGrid<3, float>* image_cartesian_ptr
          = dynamic_cast<DiscretisedDensityOnCartesianGrid<3, float>*>(((dyn_image.get_densities())[0]).get());
      const BasicCoordinate<3, float> this_grid_spacing = image_cartesian_ptr->get_grid_spacing();
      this->_model_matrix.scale_model_matrix(this_grid_spacing[2] / dyn_image.get_scanner_default_bin_size());
#ifndef NDEBUG
      this->_model_matrix.write_to_file("patlak_matrix_in_correct_scale.txt", this->_num_conv_params);
#endif // NDEBUG
    }

  this->_model_matrix.multiply_parametric_image_with_model(dyn_image, par_image, this->_num_conv_params);
}

void
GeneralizedPatlakPlot::get_generalized_patlak_parameters_from_impulse_response(
    Parametric3VoxelsOnCartesianGrid& par_image,
    const DynamicDiscretisedDensity& dyn_image,
    const DynamicDiscretisedDensity& impulse_response) const
{
  if (!this->_in_correct_scale)
    {
#ifndef NDEBUG
      this->_model_matrix.write_to_file("patlak_matrix_not_in_correct_scale.txt", this->_num_conv_params);
#endif // NDEBUG
      const DiscretisedDensityOnCartesianGrid<3, float>* image_cartesian_ptr
          = dynamic_cast<DiscretisedDensityOnCartesianGrid<3, float>*>(((dyn_image.get_densities())[0]).get());
      const BasicCoordinate<3, float> this_grid_spacing = image_cartesian_ptr->get_grid_spacing();
      this->_model_matrix.scale_model_matrix(this_grid_spacing[2] / dyn_image.get_scanner_default_bin_size());
#ifndef NDEBUG
      this->_model_matrix.write_to_file("patlak_matrix_in_correct_scale.txt", this->_num_conv_params);
#endif // NDEBUG
    }

  this->_model_matrix.estimate_generalized_patlak_parameters_with_impulse_response(
      par_image, impulse_response, this->_num_conv_params);
}

void
GeneralizedPatlakPlot::multiply_dynamic_image_with_initialization_model_gradient(Parametric3VoxelsOnCartesianGrid& par_image,
                                                                                 const DynamicDiscretisedDensity& dyn_image) const
{
  if (!this->_in_correct_scale)
    {
#ifndef NDEBUG
      this->_initialization_model_matrix.write_to_file("initialization_patlak_matrix_not_in_correct_scale.txt");
#endif // NDEBUG
      const DiscretisedDensityOnCartesianGrid<3, float>* image_cartesian_ptr
          = dynamic_cast<DiscretisedDensityOnCartesianGrid<3, float>*>(((dyn_image.get_densities())[0]).get());
      const BasicCoordinate<3, float> this_grid_spacing = image_cartesian_ptr->get_grid_spacing();
      this->_initialization_model_matrix.scale_model_matrix(this_grid_spacing[2] / dyn_image.get_scanner_default_bin_size());
#ifndef NDEBUG
      this->_initialization_model_matrix.write_to_file("initialization_patlak_matrix_in_correct_scale.txt");
#endif // NDEBUG
    }
  this->_initialization_model_matrix.multiply_dynamic_image_with_initialization_model(par_image, dyn_image);
}

void
GeneralizedPatlakPlot::multiply_dynamic_image_with_initialization_model_gradient_and_add_to_input(
    Parametric3VoxelsOnCartesianGrid& par_image, const DynamicDiscretisedDensity& dyn_image) const
{
  if (!this->_in_correct_scale)
    {
#ifndef NDEBUG
      this->_initialization_model_matrix.write_to_file("initialization_patlak_matrix_not_in_correct_scale.txt");
#endif // NDEBUG
      const DiscretisedDensityOnCartesianGrid<3, float>* image_cartesian_ptr
          = dynamic_cast<DiscretisedDensityOnCartesianGrid<3, float>*>(((dyn_image.get_densities())[0]).get());
      const BasicCoordinate<3, float> this_grid_spacing = image_cartesian_ptr->get_grid_spacing();
      this->_initialization_model_matrix.scale_model_matrix(this_grid_spacing[2] / dyn_image.get_scanner_default_bin_size());
#ifndef NDEBUG
      this->_initialization_model_matrix.write_to_file("initialization_patlak_matrix_in_correct_scale.txt");
#endif // NDEBUG
    }
  this->_initialization_model_matrix.multiply_dynamic_image_with_initialization_model_and_add_to_input(par_image, dyn_image);
}

// Should be a virtual function declared in the KineticModels or better to the LinearModels
void
GeneralizedPatlakPlot::get_dynamic_image_from_initialization_parametric_image(
    DynamicDiscretisedDensity& dyn_image, const Parametric3VoxelsOnCartesianGrid& par_image) const
{
  if (!this->_in_correct_scale)
    {
#ifndef NDEBUG
      this->_initialization_model_matrix.write_to_file("patlak_matrix_not_in_correct_scale.txt");
#endif // NDEBUG
      const DiscretisedDensityOnCartesianGrid<3, float>* image_cartesian_ptr
          = dynamic_cast<DiscretisedDensityOnCartesianGrid<3, float>*>(((dyn_image.get_densities())[0]).get());
      const BasicCoordinate<3, float> this_grid_spacing = image_cartesian_ptr->get_grid_spacing();
      this->_initialization_model_matrix.scale_model_matrix(this_grid_spacing[2] / dyn_image.get_scanner_default_bin_size());
#ifndef NDEBUG
      this->_initialization_model_matrix.write_to_file("initialization_patlak_matrix_in_correct_scale.txt");
#endif // NDEBUG
    }

  this->_initialization_model_matrix.multiply_parametric_image_with_initialization_model(dyn_image, par_image);
}

void
GeneralizedPatlakPlot::estimate_nested_loop_parameters_with_model(Parametric3VoxelsOnCartesianGrid& parametric_image,
                                                                  DynamicDiscretisedDensity& dynamic_image_nested_loop_estimate,
                                                                  DynamicDiscretisedDensity& dynamic_image_update_factor,
                                                                  const DynamicDiscretisedDensity& dynamic_image_reference,
                                                                  float minimum_nested_relative_change,
                                                                  float maximum_nested_relative_change,
                                                                  int num_nested_subiterations) const
{
  if (!this->_in_correct_scale)
    {
#ifndef NDEBUG
      this->_model_matrix.write_to_file("patlak_matrix_not_in_correct_scale.txt", this->_num_conv_params);
#endif // NDEBUG
      const DiscretisedDensityOnCartesianGrid<3, float>* image_cartesian_ptr
          = dynamic_cast<DiscretisedDensityOnCartesianGrid<3, float>*>(
              ((dynamic_image_nested_loop_estimate.get_densities())[0]).get());
      const BasicCoordinate<3, float> this_grid_spacing = image_cartesian_ptr->get_grid_spacing();
      this->_model_matrix.scale_model_matrix(this_grid_spacing[2]
                                             / dynamic_image_nested_loop_estimate.get_scanner_default_bin_size());
#ifndef NDEBUG
      this->_model_matrix.write_to_file("patlak_matrix_in_correct_scale.txt", this->_num_conv_params);
#endif // NDEBUG
    }
  this->_model_matrix.estimate_nested_loop_parameters_with_model(parametric_image,
                                                                 dynamic_image_nested_loop_estimate,
                                                                 dynamic_image_update_factor,
                                                                 dynamic_image_reference,
                                                                 num_nested_subiterations,
                                                                 minimum_nested_relative_change,
                                                                 maximum_nested_relative_change,
                                                                 this->_num_conv_params);
}

unsigned int
GeneralizedPatlakPlot::get_num_conv_params() const
{
  return this->_num_conv_params;
}

void
GeneralizedPatlakPlot::initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("Generalized Patlak Plot Parameters");
  this->parser.add_key("convolution sampling interval", &this->_conv_sample_interval);
  this->parser.add_key("kloss lower bound", &this->_kloss_lb);
  this->parser.add_key("kloss upper bound", &this->_kloss_ub);
  this->parser.add_key("number of kloss samples", &this->_kloss_num_samples);
  this->parser.add_stop_key("end Generalized Patlak Plot Parameters");
}

/*! \todo This currently hard-wired F-18 decay for the plasma data */
bool
GeneralizedPatlakPlot::post_processing()
{
  if (base_type::post_processing() == true)
    return true;

  this->_plasma_frame_data.read_plasma_data(this->_blood_data_filename); // The implementation assumes three list file.
  // TODO have parameter
  warning("Assuming F-18 tracer for plasma data!!!");
  this->_plasma_frame_data.set_isotope_halflife(6586.2F);
  this->_plasma_frame_data.shift_time(this->_time_shift);

  // this->_plasma_frame_data = this->_complete_plasma_data.get_sample_data_in_frames(this->_frame_defs);
  this->_num_frames = this->_plasma_frame_data.size();
  float _last_frame_time = this->get_frame_reference_time() == 1
                               ? floor(0.5
                                       * (this->get_time_frame_definitions().get_end_time(this->_num_frames)
                                          + this->get_time_frame_definitions().get_start_time(this->_num_frames)))
                               : this->get_time_frame_definitions().get_end_time(this->_num_frames);
  this->_last_frame_ref_time = static_cast<unsigned int>(floor(_last_frame_time + 0.5));
  this->_num_conv_params = ((this->_last_frame_ref_time - 1) / this->_conv_sample_interval) + 2;

  return false;
}

END_NAMESPACE_STIR