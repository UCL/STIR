/*
    Copyright (C) 2020 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

/*!
  \file
  \ingroup recon_buildblock
  \brief Declaration of class stir::SqrtHessianRowSum

  \author Robert Twyman
  \author Kris Thielemans
*/

#include "stir/recon_buildblock/SqrtHessianRowSum.h"
#include "stir/info.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/IO/read_from_file.h"
#include "stir/is_null_ptr.h"
#include "stir/recon_buildblock/GeneralisedObjectiveFunction.h"
#include "stir/unique_ptr.h"


START_NAMESPACE_STIR

template <typename TargetT>
SqrtHessianRowSum<TargetT>::
SqrtHessianRowSum()
{
  this->set_defaults();
}

template <typename TargetT>
SqrtHessianRowSum<TargetT>::
SqrtHessianRowSum(const std::string& filename)
{
  this->set_defaults();
  this->parse(filename.c_str());
}

template <typename TargetT>
void
SqrtHessianRowSum<TargetT>::
set_defaults()
{
  output_file_format_sptr = OutputFileFormat<TargetT>::default_sptr();
  output_filename = "";
  set_use_approximate_hessian(true);
  set_compute_with_penalty(false);

  _already_setup = false;
}

template <typename TargetT>
void
SqrtHessianRowSum<TargetT>::initialise_keymap()
{
  parser.add_start_key("SqrtHessianRowSum Parameters");
  parser.add_key("output filename", &output_filename);
  parser.add_key("input image filename", &input_image_filename);
  parser.add_key("use approximate Hessian", &use_approximate_hessian);
  parser.add_key("compute with penalty", &compute_with_penalty);
  parser.add_parsing_key("objective function type", &objective_function_sptr);
  parser.add_stop_key("End");
}

template <typename TargetT>
bool
SqrtHessianRowSum<TargetT>::post_processing()
{

  if (input_image_filename.empty())
  {
    error("Please define input_image_filename.");
    return true;
  }
  input_image_sptr = read_from_file<TargetT>(input_image_filename);
  return false;
}

template <typename TargetT>
GeneralisedObjectiveFunction<TargetT > const&
SqrtHessianRowSum<TargetT>::
get_objective_function_sptr()
{
  return static_cast<GeneralisedObjectiveFunction<TargetT >&> (*objective_function_sptr);
}

template <typename TargetT>
void
SqrtHessianRowSum<TargetT>::
set_objective_function_sptr(const shared_ptr<GeneralisedObjectiveFunction<TargetT > >& obj_fun)
{
  this->objective_function_sptr  = obj_fun;
  // it might be that it's already set-up, but we don't know
  _already_setup = false;
}

template <typename TargetT>
shared_ptr <TargetT>
SqrtHessianRowSum<TargetT>::
get_input_image_sptr()
{
  return input_image_sptr;
}

template <typename TargetT>
void
SqrtHessianRowSum<TargetT>::
set_input_image_sptr(shared_ptr <TargetT > const& image_sptr)
{
  if (_already_setup)
    _already_setup = input_image_sptr->has_same_characteristics(*image_sptr);
  input_image_sptr = image_sptr;
}

template <typename TargetT>
shared_ptr <TargetT>
SqrtHessianRowSum<TargetT>::
get_output_target_sptr()
{
  return output_target_sptr;
}

template <typename TargetT>
bool
SqrtHessianRowSum<TargetT>::
get_use_approximate_hessian() const
{
  return use_approximate_hessian;
}

template <typename TargetT>
void
SqrtHessianRowSum<TargetT>::
set_use_approximate_hessian(bool use_approximate)
{
  use_approximate_hessian = use_approximate;
}

template <typename TargetT>
bool
SqrtHessianRowSum<TargetT>::
get_compute_with_penalty() const
{
  return compute_with_penalty;
}

template <typename TargetT>
void
SqrtHessianRowSum<TargetT>::
set_compute_with_penalty(bool with_penalty)
{
  compute_with_penalty = with_penalty;
}

template <typename TargetT>
void
SqrtHessianRowSum<TargetT>::
set_up()
{
  if (is_null_ptr(this->objective_function_sptr))
  {
    error("objective_function_sptr is null");
  }
  if (is_null_ptr(this->input_image_sptr))
  {
    error("input_image_sptr is null");
  }
  objective_function_sptr->set_up(input_image_sptr);
  output_target_sptr =  unique_ptr<TargetT>(input_image_sptr->get_empty_copy());
  std::fill(output_target_sptr->begin_all(), output_target_sptr->end_all(), 0.F);

  _already_setup = true;
}

template <typename TargetT>
void
SqrtHessianRowSum<TargetT>::
process_data()
{
  if (get_use_approximate_hessian())
  {
    // Compute the SqrtHessianRowSum image using the approximate hessian
    // The input image is used as a template for this method
    compute_approximate_Hessian_row_sum();
  }
  else
  {
    // Compute the SqrtHessianRowSum image with the full Hessian at the input image estimate.
    // The input image here is assumed to be the current_image_estimate at the point the Hessian will be computed
    compute_Hessian_row_sum();
  }

  // Square Root the output of the Hessian_row_sum methods
  std::for_each(output_target_sptr->begin_all(), output_target_sptr->end_all(),
                [](float& a) { return a=sqrt(a); } );

  // Save the output
  if (!output_filename.empty())
    {
      output_file_format_sptr->write_to_file(output_filename, *output_target_sptr);
      info("Output image of sqrt Hessian row sum has been computed and saved as " + output_filename + ".");
    }
}

template <typename TargetT>
void
SqrtHessianRowSum<TargetT>::
compute_Hessian_row_sum()
{
  if (!_already_setup)
    error("set_up() needs to be called first");

  auto ones_image_sptr = input_image_sptr->get_empty_copy();
  std::fill(ones_image_sptr->begin_all(), ones_image_sptr->end_all(), 1.F);

  info("Computing Hessian row sum, this may take a while...");
  if (get_compute_with_penalty())
    objective_function_sptr->accumulate_Hessian_times_input(*output_target_sptr, *input_image_sptr, *ones_image_sptr);
  else
    objective_function_sptr->accumulate_Hessian_times_input_without_penalty(*output_target_sptr, *input_image_sptr, *ones_image_sptr);
}

template <typename TargetT>
void
SqrtHessianRowSum<TargetT>::
compute_approximate_Hessian_row_sum()
{
  if (!_already_setup)
    error("set_up() needs to be called first");

  output_target_sptr =  unique_ptr<TargetT>(input_image_sptr->get_empty_copy());
  std::fill(output_target_sptr->begin_all(), output_target_sptr->end_all(), 0.F);
  info("Computing the approximate Hessian row sum, this may take a while...");
  // Setup image
  auto ones_image_sptr = input_image_sptr->get_empty_copy();
  std::fill(ones_image_sptr->begin_all(), ones_image_sptr->end_all(), 1.F);

  // Approximate Hessian computation will error for a lot of priors so we ignore it!
  if (get_compute_with_penalty())
    info("approximate Hessian row sum: Priors do not have an approximation of the Hessian. Ignoring the prior!");

  objective_function_sptr->add_multiplication_with_approximate_Hessian_without_penalty(*output_target_sptr,
                                                                                       *ones_image_sptr);
}

template class SqrtHessianRowSum<DiscretisedDensity<3,float> >;
template class SqrtHessianRowSum<ParametricVoxelsOnCartesianGrid >;
//template class SqrtHessianRowSum<GatedDiscretisedDensity>;
END_NAMESPACE_STIR
