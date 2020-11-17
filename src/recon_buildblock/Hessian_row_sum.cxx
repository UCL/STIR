/*
    Copyright (C) 2020 University College London
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    See STIR/LICENSE.txt for details
*/

/*!
  \file
  \ingroup recon_buildblock
  \brief Declaration of class stir::Hessian_row_sum

  \author Robert Twyman
  \author Kris Thielemans
*/

#include "stir/recon_buildblock/Hessian_row_sum.h"
#include "stir/info.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/IO/read_from_file.h"
#include "stir/is_null_ptr.h"
#include "stir/recon_buildblock/GeneralisedObjectiveFunction.h"
//#include "stir/recon_buildblock/GeneralisedPrior.h"
//#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h"
#include "stir/unique_ptr.h"


START_NAMESPACE_STIR

template <typename TargetT>
Hessian_row_sum<TargetT>::
Hessian_row_sum()
{
  this->set_defaults();
}

template <typename TargetT>
void
Hessian_row_sum<TargetT>::
set_defaults()
{
  output_file_format_sptr = OutputFileFormat<TargetT>::default_sptr();
  output_filename = "SpatiallyVariantPenaltyStrength.hv";
  set_use_approximate_hessian(true);
  set_compute_with_penalty(false);
}

template <typename TargetT>
void
Hessian_row_sum<TargetT>::initialise_keymap()
{
  parser.add_start_key("Hessian Row Sum Computation Parameters");
  parser.add_key("output filename", &output_filename);
  parser.add_key("input image filename", &input_image_filename);
  parser.add_key("use approximate hessian", &use_approximate_hessian);
  parser.add_key("compute with penalty", &compute_with_penalty);
  parser.add_parsing_key("objective function type", &objective_function_sptr);
  parser.add_stop_key("End");
}

template <typename TargetT>
bool
Hessian_row_sum<TargetT>::post_processing()
{
  if (is_null_ptr(this->objective_function_sptr))
  {
    error("objective_function_sptr is null");
    return true;
  }

  if (input_image_filename.empty())
  {
    error("Please define input_image_filename.");
    return true;
  }
  return false;
}

template <typename TargetT>
GeneralisedObjectiveFunction<TargetT > const&
Hessian_row_sum<TargetT>::
get_objective_function()
{
  return static_cast<GeneralisedObjectiveFunction<TargetT >&> (*objective_function_sptr);
}

template <typename TargetT>
void
Hessian_row_sum<TargetT>::
set_objective_function_sptr(const shared_ptr<GeneralisedObjectiveFunction<TargetT > >& obj_fun)
{
  this->objective_function_sptr  = obj_fun;
}

template <typename TargetT>
shared_ptr <TargetT>
Hessian_row_sum<TargetT>::
get_input_image()
{
  return input_image;
}

template <typename TargetT>
void
Hessian_row_sum<TargetT>::
set_input_image(shared_ptr <TargetT > const& image)
{
  input_image = image;
}

template <typename TargetT>
shared_ptr <TargetT>
Hessian_row_sum<TargetT>::
get_output_target_sptr()
{
  return output_target_sptr;
}

template <typename TargetT>
void
Hessian_row_sum<TargetT>::
reset_output_target_sptr(shared_ptr <TargetT > const& image)
{
  output_target_sptr =  unique_ptr<TargetT>(image->get_empty_copy());
  std::fill(output_target_sptr->begin_all(), output_target_sptr->end_all(), 0.F);
}

template <typename TargetT>
bool
Hessian_row_sum<TargetT>::
get_use_approximate_hessian()
{
  return use_approximate_hessian;
}

template <typename TargetT>
void
Hessian_row_sum<TargetT>::
set_use_approximate_hessian(bool use_approximate)
{
  use_approximate_hessian = use_approximate;
}

template <typename TargetT>
bool
Hessian_row_sum<TargetT>::
get_compute_with_penalty()
{
  return compute_with_penalty;
}

template <typename TargetT>
void
Hessian_row_sum<TargetT>::
set_compute_with_penalty(bool with_penalty)
{
  compute_with_penalty = with_penalty;
}

template <typename TargetT>
void
Hessian_row_sum<TargetT>::
process_data()
{
  input_image = read_from_file<TargetT>(input_image_filename);
  objective_function_sptr->set_up(input_image);

  if (get_use_approximate_hessian())
  {
    // Compute the Hessian_row_sum image using the approximate hessian
    // The input image is used as a template for this method
    compute_approximate_Hessian_row_sum();
  }
  else
  {
    // Compute the Hessian_row_sum image with the full hessian at the input image estimate.
    // The input image here is assumed to be the current_image_estimate at the point the hessian will be computed
    compute_Hessian_row_sum();
  }

  // Spatially variant penalty strength is defined as the sqrt of the output of Hessian_row_sum methods
  std::for_each(output_target_sptr->begin_all(), output_target_sptr->end_all(),
                [](float& a) { return a=sqrt(a); } );
  // Save the output
  output_file_format_sptr->write_to_file(output_filename, *output_target_sptr);
  info("Spatially variant penalty strength has been computed and saved as " + output_filename + ".");
}

template <typename TargetT>
void
Hessian_row_sum<TargetT>::
compute_Hessian_row_sum()
{
  reset_output_target_sptr(input_image);
  auto ones_image_sptr = input_image->get_empty_copy();
  std::fill(ones_image_sptr->begin_all(), ones_image_sptr->end_all(), 1.F);

  info("Computing Hessian row sum, this may take a while...");
  if (get_compute_with_penalty())
    objective_function_sptr->accumulate_Hessian_times_input(*output_target_sptr, *input_image, *ones_image_sptr);
  else
    objective_function_sptr->accumulate_Hessian_times_input_without_penalty(*output_target_sptr, *input_image, *ones_image_sptr);
}

template <typename TargetT>
void
Hessian_row_sum<TargetT>::
compute_approximate_Hessian_row_sum()
{
  reset_output_target_sptr(input_image);
  info("Computing the approximate Hessian row sum, this may take a while...");
  // Setup image
  auto ones_image_sptr = input_image->get_empty_copy();
  std::fill(ones_image_sptr->begin_all(), ones_image_sptr->end_all(), 1.F);

  // Approximate Hessian computation will error for a lot of priors so we ignore it!
  if (get_compute_with_penalty())
    info("approximate Hessian row sum: Priors do not have an approximation of the Hessian. Ignoring the prior!");

  objective_function_sptr->add_multiplication_with_approximate_Hessian_without_penalty(*output_target_sptr,
                                                                                       *ones_image_sptr);
}

template class Hessian_row_sum<DiscretisedDensity<3,float> >;
template class Hessian_row_sum<ParametricVoxelsOnCartesianGrid >;
//template class Hessian_row_sum<GatedDiscretisedDensity>;
END_NAMESPACE_STIR
