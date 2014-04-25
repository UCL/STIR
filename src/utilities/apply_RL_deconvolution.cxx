//
/*
 Copyright (C) 2009 - 2013, King's College London
 This file is part of STIR.

 This file is free software; you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation; either version 2.3 of the License, or
 (at your option) any later version.

 This file is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Lesser General Public License for more details.

 See STIR/LICENSE.txt for details
 */
/*!
 \file
 \ingroup utilities
 \ingroup spatial_transformation

 \brief This program applies motion transformation (warping) and then either accumulates or averages the resulting gated images
 \author Nicolas A Karakatsanis
 */
#include "stir/IO/OutputFileFormat.h"
#include "stir/IO/read_from_file.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/DiscretisedDensity.h"
#include "stir/GatedDiscretisedDensity.h"
#include "stir/spatial_transformation/GatedSpatialTransformation.h"
#include "stir/Succeeded.h"

#include "stir/Viewgram.h"
#include "stir/RelatedViewgrams.h"
#include "stir/stream.h"
#include "stir/recon_array_functions.h"
#include "stir/is_null_ptr.h"
#include "stir/numerics/divide.h"
#include "stir/thresholding.h"
#include "stir/NumericInfo.h"

#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <string.h>

#include "stir/CPUTimer.h"
#include "stir/info.h"
#include <boost/format.hpp>

#ifndef STIR_NO_NAMESPACES
using std::ends;
using std::cerr;
using std::endl;
using std::max;
using std::string;
#endif

USING_NAMESPACE_STIR

using namespace BSpline;

const float small_num = 0.000001F;

static void
print_usage_and_exit()
{
  cerr << "\nUsage: apply_RL_deconvolution <output filename> <input filename> <forward motion vectors prefix> <reverse motion "
          "vectors prefix> <number of EM iterations> <save iteration interval>\n"
       << "\t--Applies Richardson-Lucy iterative deconvolution to a motion-contaminated image utilizing the user-specified "
          "forward and backward-motion vector fields \n"
       << "\t--<forward motion vectors prefix>: Filename prefix indicating the files of the motion vector fields used to "
          "introduce motion contamination to the original motion-less image "
       << "\t--<reverse motion vectors prefix>: Filename prefix indicating the files of the motion vector fields used to reverse "
          "the motion contamination (flipped motion kernel) to the motion-contaminated image "
       << "\t--<save iteration interval>: Integer designating the interval of Richardson-Lucy EM iterations at which to save "
          "images every time";
  exit(EXIT_FAILURE);
}

void
RL_deconvolution(DiscretisedDensity<3, float>& gradient,
                 DiscretisedDensity<3, float>& current_estimate,
                 DiscretisedDensity<3, float>& model_sensitivity_image,
                 DiscretisedDensity<3, float>& conv_image_estimate,
                 DiscretisedDensity<3, float>& conv_image_RL_estimate,
                 const DiscretisedDensity<3, float>& conv_image_reference,
                 GatedSpatialTransformation motion_vectors,
                 GatedSpatialTransformation reverse_motion_vectors,
                 const unsigned int num_iterations,
                 const unsigned int save_iterations_interval,
                 float min_update_factor,
                 float max_update_factor,
                 const char* output_filename)
{

  // Initializations-Declarations
  std::stringstream iter_num_str;

  // First, pre-compute motion model sensitivity image for proper Richardson-Lucy EM deconvolution

  // Initialize model sensitivity image and convolved image of ONES
  std::fill(model_sensitivity_image.begin_all(), model_sensitivity_image.end_all(), 1.F);

  shared_ptr<DiscretisedDensity<3, float>> conv_image_of_all_ones(conv_image_reference.get_empty_copy());
  std::fill(conv_image_of_all_ones->begin_all(), conv_image_of_all_ones->end_all(), 1.F);

  cerr << "Computing model sensitivity image..." << endl;

  // To obtain motion model sensitivity image reversely translate (correct-for-motion or equivalent to back project operation for
  // motion model) from motion-gated space to motion-corrected single-gate space using a gated image estimate of ALL ONES
  // reverse_motion_vectors.average_warp_image(model_sensitivity_image,
  //                                          *conv_image_of_all_ones);
  //

  // Print out the min and max values of the model sensitivity image
  const float current_min_model_sensitivity
      = *std::min_element(model_sensitivity_image.begin_all(), model_sensitivity_image.end_all());
  const float current_max_model_sensitivity
      = *std::max_element(model_sensitivity_image.begin_all(), model_sensitivity_image.end_all());
  cerr << "Model sensitivity image "
       << ", (min, max): (" << current_min_model_sensitivity << ", " << current_max_model_sensitivity << ")" << endl;

  cerr << "Model sensitivity image has been computed." << endl;

  // Print out the min and max values of the convolved image reference
  const float current_min_reference = *std::min_element(conv_image_reference.begin_all(), conv_image_reference.end_all());
  const float current_max_reference = *std::max_element(conv_image_reference.begin_all(), conv_image_reference.end_all());
  cerr << "Reference convolved image , (min, max): (" << current_min_reference << ", " << current_max_reference << ")" << endl
       << endl;

  // Then, enter EM loop of Richardson-Lucy deconvolution of motion contaminated input image (conv_image_reference) utilizing the
  // previous sensitivity image
  cerr << endl
       << "Entering EM update loop (" << num_iterations << " Richardson-Lucy EM deconvolution iterations for motion correction)."
       << endl;

  for (unsigned int iterations_num = 1; iterations_num <= num_iterations; iterations_num++)
    {

      // Print out the min and max values of the initial deconvolved image estimate
      const float current_min_estimate = *std::min_element(current_estimate.begin_all(), current_estimate.end_all());
      const float current_max_estimate = *std::max_element(current_estimate.begin_all(), current_estimate.end_all());
      cerr << "Richardson-Lucy iteration: " << iterations_num << " Initial deconvolved image estimate , (min, max): ("
           << current_min_estimate << ", " << current_max_estimate << ")" << endl;

      // Translate (contaminate-with-motion or equivalent to convolve or forward project operation for motion model)
      // from deconvolved (motion-corrected) image space to convolved (motion-contaminated) image space using a motion-corrected
      // image estimate as an input
      motion_vectors.average_warp_image(conv_image_RL_estimate, current_estimate);

      // Print out the min and max values of the current convolved image estimate after the forward convolution
      const float conv_min_RL_estimate = *std::min_element(conv_image_RL_estimate.begin_all(), conv_image_RL_estimate.end_all());
      const float conv_max_RL_estimate = *std::max_element(conv_image_RL_estimate.begin_all(), conv_image_RL_estimate.end_all());
      cerr << "Richardson-Lucy iteration: " << iterations_num
           << " Current convolved image estimate after forward convolution , (min, max): (" << conv_min_RL_estimate << ", "
           << conv_max_RL_estimate << ")" << endl;

      conv_image_estimate = conv_image_reference;

      // Division of the reference with the estimate in the convolved space
      divide(conv_image_estimate.begin_all(), conv_image_estimate.end_all(), conv_image_RL_estimate.begin_all(), small_num);

      // Print out the min and max values of the convolved image gradient after the division of the reference with the estimate in
      // the convolved space
      const float conv_min_estimate = *std::min_element(conv_image_estimate.begin_all(), conv_image_estimate.end_all());
      const float conv_max_estimate = *std::max_element(conv_image_estimate.begin_all(), conv_image_estimate.end_all());
      cerr << "Richardson-Lucy iteration: " << iterations_num
           << " Current convolved image gradient after division of reference with the estimate in the convolved space , (min, "
              "max): ("
           << conv_min_estimate << ", " << conv_max_estimate << ") , division small number: " << small_num << endl;

      // Reversely translate (correct-for-motion or deconvolve equivalent to back project operation for motion model)
      // from convolved (motion-contaminated) image space to deconvolved (motion-corrected) image space using the convolved image
      // estimate as an input
      reverse_motion_vectors.average_warp_image(gradient, conv_image_estimate);

      // Print out the min and max values of the deconvolved image gradient after the backward convolution of the previous
      // gradient from the convolved space
      const float unnormalized_gradient_min_estimate = *std::min_element(gradient.begin_all(), gradient.end_all());
      const float unnormalized_gradient_max_estimate = *std::max_element(gradient.begin_all(), gradient.end_all());
      cerr << "Richardson-Lucy iteration: " << iterations_num
           << " Current deconvolved image gradient after the backward convolution of the previous image gradient from the "
              "convolved space , (min, max): ("
           << unnormalized_gradient_min_estimate << ", " << unnormalized_gradient_max_estimate << ")" << endl;

      // Divide/Normalize by motion model sensitivity image
      // divide(gradient.begin_all(),
      //       gradient.end_all(),
      //  	   model_sensitivity_image.begin_all(),
      //       small_num);

      /* Include this for testing purposes
      if ( (iterations_num % save_iterations_interval)==0 )
    {
         iter_num_str.str(string());
     iter_num_str << "_norm_bwdconv_grad_it" << iterations_num;
     string iter_output_filename = output_filename + iter_num_str.str();

         OutputFileFormat<DiscretisedDensity<3,float> >::
       default_sptr()-> write_to_file(iter_output_filename, gradient);
     }
      */

      // Print out the min and max values of the normalized deconvolved image gradient after the devision to the model sensitivity
      // image
      const float current_min_gradient = *std::min_element(gradient.begin_all(), gradient.end_all());
      const float current_max_gradient = *std::max_element(gradient.begin_all(), gradient.end_all());
      cerr << "Richardson-Lucy iteration: " << iterations_num
           << " Gradient after sensitivity image division: old value (min, max): (" << current_min_gradient << ", "
           << current_max_gradient << "), new value (min, max) (" << max(current_min_gradient, min_update_factor) << ", "
           << min(current_max_gradient, max_update_factor) << ")"
           << "), threshold limits (min, max) (" << min_update_factor << ", " << max_update_factor << ")" << endl;

      zero_threshold_upper_lower(gradient.begin_all(), gradient.end_all(), min_update_factor, max_update_factor);

      // EM updates of motion-corrected image estimates
      {
        DiscretisedDensity<3, float>::const_full_iterator gradient_iter = gradient.begin_all_const();
        const DiscretisedDensity<3, float>::const_full_iterator end_gradient_iter = gradient.end_all_const();
        DiscretisedDensity<3, float>::full_iterator current_estimate_iter = current_estimate.begin_all();
        while (gradient_iter != end_gradient_iter)
          {
            *current_estimate_iter *= (*gradient_iter);
            ++current_estimate_iter;
            ++gradient_iter;
          }
      }

      // Print out the min and max values of the nested updated image for each nested iteration
      const float current_min_updated_image = *std::min_element(current_estimate.begin_all(), current_estimate.end_all());
      const float current_max_updated_image = *std::max_element(current_estimate.begin_all(), current_estimate.end_all());
      cerr << "Richardson-Lucy iteration: " << iterations_num
           << " Updated deconvolved (motion corrected) image value (min, max) (" << current_min_updated_image << ", "
           << current_max_updated_image << ")" << endl
           << endl;

      if ((iterations_num % save_iterations_interval) == 0)
        {
          iter_num_str.str(string());
          iter_num_str << "_it" << iterations_num;
          string iter_output_filename = output_filename + iter_num_str.str();

          Succeeded write_result = OutputFileFormat<DiscretisedDensity<3, float>>::default_sptr()->write_to_file(
              iter_output_filename, current_estimate);

          if (write_result == Succeeded::yes)
            cerr << "Richardson-Lucy iteration: " << iterations_num
                 << " Writing of current deconvolved image estimate (header file " << iter_output_filename << ") was successful."
                 << endl
                 << endl
                 << endl;
          else
            {
              cerr << "Richardson-Lucy iteration: " << iterations_num
                   << " Writing of current deconvolved image estimate (header file " << iter_output_filename
                   << ") did not succeed." << endl
                   << endl
                   << endl;
              exit(EXIT_FAILURE);
            }
        }
    }

  cerr << "End of EM deconvolution process to compute motion-corrected estimates (after " << num_iterations
       << " Richardson-Lucy iterations)" << endl
       << endl;
}

int
main(int argc, char** argv)
{
  if (argc != 7)
    print_usage_and_exit();

  // get parameters from command line
  char const* const output_filename = argv[1];
  char const* const input_filename = argv[2];

  float min_update_factor = 0;
  float max_update_factor = NumericInfo<float>().max_value() - 1;

  cerr << "\nPost-reconstruction application of Richardson-Lucy EM deconvolution algorithm to correct for intra-frame/gate motion"
       << endl
       << endl;

  const shared_ptr<DiscretisedDensity<3, float>> convolved_density_sptr(
      read_from_file<DiscretisedDensity<3, float>>(input_filename));

  GatedSpatialTransformation motion_vectors;
  GatedSpatialTransformation reverse_motion_vectors;

  motion_vectors.read_from_files(argv[3]);
  reverse_motion_vectors.read_from_files(argv[4]);

  const unsigned int num_iterations(atoi(argv[5]));
  const unsigned int save_iterations_interval(atoi(argv[6]));

  cerr << "\nNumber of Richardson-Lucy deconvolution iterations: " << num_iterations << endl
       << "\nSave images every " << save_iterations_interval << " iterations" << endl
       << endl
       << "\nLow threshold for the update factors: " << min_update_factor << endl
       << "Upper threshold for the update factors: " << max_update_factor << endl
       << endl;

  shared_ptr<DiscretisedDensity<3, float>> corrected_density_sptr(convolved_density_sptr->get_empty_copy());
  shared_ptr<DiscretisedDensity<3, float>> gradient_sptr(convolved_density_sptr->get_empty_copy());
  shared_ptr<DiscretisedDensity<3, float>> model_sensitivity_image_sptr(convolved_density_sptr->get_empty_copy());
  shared_ptr<DiscretisedDensity<3, float>> convolved_image_estimate_sptr(convolved_density_sptr->get_empty_copy());
  shared_ptr<DiscretisedDensity<3, float>> convolved_image_RL_estimate_sptr(convolved_density_sptr->get_empty_copy());

  // Initialize the deconvolved estimate with an image of ONES for the first EM iteration
  std::fill(corrected_density_sptr->begin_all(), corrected_density_sptr->end_all(), 1.F);

  RL_deconvolution(*gradient_sptr,
                   *corrected_density_sptr,
                   *model_sensitivity_image_sptr,
                   *convolved_image_estimate_sptr,
                   *convolved_image_RL_estimate_sptr,
                   *convolved_density_sptr,
                   motion_vectors,
                   reverse_motion_vectors,
                   num_iterations,
                   save_iterations_interval,
                   min_update_factor,
                   max_update_factor,
                   output_filename);

  string sensitivity_prefix = "sens_";
  string output_ending = "_final";
  string sensitivity_image_filename = sensitivity_prefix + output_filename;
  string final_iter_image_filename = output_filename + output_ending;

  OutputFileFormat<DiscretisedDensity<3, float>>::default_sptr()->write_to_file(sensitivity_image_filename,
                                                                                *model_sensitivity_image_sptr);
  const Succeeded res = OutputFileFormat<DiscretisedDensity<3, float>>::default_sptr()->write_to_file(final_iter_image_filename,
                                                                                                      *corrected_density_sptr);

  return res == Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}