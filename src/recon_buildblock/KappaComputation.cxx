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
  \brief Declaration of class stir::KappaComputation

  \author Robert Twyman
  \author Kris Thielemans
*/

#include "stir/recon_buildblock/KappaComputation.h"
#include "stir/info.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/IO/read_from_file.h"
#include "stir/is_null_ptr.h"
//#include "stir/recon_buildblock/GeneralisedObjectiveFunction.h"
//#include "stir/recon_buildblock/GeneralisedPrior.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h"
#include "stir/unique_ptr.h"


START_NAMESPACE_STIR

template <typename TargetT>
KappaComputation<TargetT>::
KappaComputation()
{
  this->set_defaults();
}

template <typename TargetT>
void
KappaComputation<TargetT>::
set_defaults()
{
  objective_function_sptr.reset(new PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>);
  output_file_format_sptr = OutputFileFormat<TargetT>::default_sptr();
  set_use_approximate_hessian(true);
  set_compute_with_penalty(false);
}

template <typename TargetT>
void
KappaComputation<TargetT>::initialise_keymap()
{
  parser.add_start_key("Kappa Computation Parameters");
  parser.add_key("kappa filename", &kappa_filename);
  parser.add_key("input image filename", &input_image_filename);
  parser.add_key("use approximate hessian", &use_approximate_hessian);
  parser.add_key("compute with penalty", &compute_with_penalty);
  parser.add_parsing_key("objective function type", &objective_function_sptr);
  parser.add_stop_key("End");
}

template <typename TargetT>
bool
KappaComputation<TargetT>::post_processing()
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
PoissonLogLikelihoodWithLinearModelForMean<TargetT > const&
KappaComputation<TargetT>::
get_objective_function()
{
  return static_cast<PoissonLogLikelihoodWithLinearModelForMean<TargetT >&> (*objective_function_sptr);
}

template <typename TargetT>
void
KappaComputation<TargetT>::
set_objective_function_sptr(const shared_ptr<PoissonLogLikelihoodWithLinearModelForMean<TargetT > >& obj_fun)
{
  this->objective_function_sptr  = obj_fun;
}

template <typename TargetT>
shared_ptr <TargetT>
KappaComputation<TargetT>::
get_input_image()
{
  return input_image;
}

template <typename TargetT>
void
KappaComputation<TargetT>::
set_input_image(shared_ptr <TargetT > const& image)
{
  input_image = image;
}

template <typename TargetT>
shared_ptr <TargetT>
KappaComputation<TargetT>::
get_kappa_image_target_sptr()
{
  return kappa_image_target_sptr;
}

template <typename TargetT>
void
KappaComputation<TargetT>::
reset_kappa_image_target_sptr(shared_ptr <TargetT > const& image)
{
  kappa_image_target_sptr =  unique_ptr<TargetT>(image->get_empty_copy());
  kappa_image_target_sptr->fill(0);
}

template <typename TargetT>
bool
KappaComputation<TargetT>::
get_use_approximate_hessian()
{
  return use_approximate_hessian;
}

template <typename TargetT>
void
KappaComputation<TargetT>::
set_use_approximate_hessian(bool use_approximate)
{
  use_approximate_hessian = use_approximate;
}

template <typename TargetT>
bool
KappaComputation<TargetT>::
get_compute_with_penalty()
{
  return compute_with_penalty;
}

template <typename TargetT>
void
KappaComputation<TargetT>::
set_compute_with_penalty(bool with_penalty)
{
  compute_with_penalty = with_penalty;
}

template <typename TargetT>
void
KappaComputation<TargetT>::
process_data()
{
  input_image = read_from_file<TargetT>(input_image_filename);
  // Unfortunately we have to setup. Therefore we will cheat with regards to sensitivities that are unused in the kappa
  // computation.
  // To prevent sensitivity from being computed (which takes a lot of computation), the sensitivity filename is set to
  // a filename known to exist, one of the file inputs, and re-computation is turned off.
  objective_function_sptr->set_recompute_sensitivity(false);
  objective_function_sptr->set_use_subset_sensitivities(false);
  objective_function_sptr->set_sensitivity_filename(input_image_filename);
  objective_function_sptr->set_up(input_image);

  if (get_use_approximate_hessian())
  {
    // Compute the kappa image using the approximate hessian
    // The input image is used as a template for this method
    compute_kappa_with_approximate();
  }
  else
  {
    //Compute the kappa image with the full hessian at the current image estimate.
    // The input image here is assumed to be the current_image_estimate at the point the hessian will be computed
    compute_kappa_at_current_image_estimate();
  }

  // Save the output
  output_file_format_sptr->write_to_file(kappa_filename, *kappa_image_target_sptr);

  info("Spatially variant penalty strength (Kappa) has been computed and saved as " + kappa_filename + ".");
}

template <typename TargetT>
void
KappaComputation<TargetT>::
compute_kappa_at_current_image_estimate()
{
  reset_kappa_image_target_sptr(input_image);
  info("Computing the spatially variant penalty strength at the current image estimate, this may take a while.");
  auto ones_image_sptr = input_image->get_empty_copy();
  ones_image_sptr->fill(1);

  if (get_compute_with_penalty())
    objective_function_sptr->accumulate_Hessian_times_input(*kappa_image_target_sptr, *input_image, *ones_image_sptr);
  else
    objective_function_sptr->accumulate_Hessian_times_input_without_penalty(*kappa_image_target_sptr, *input_image, *ones_image_sptr);

  // Kappa is defined as the sqrt of the output of accumulate_Hessian_times_input
  sqrt_image(*kappa_image_target_sptr);
}

template <typename TargetT>
void
KappaComputation<TargetT>::
compute_kappa_with_approximate()
{
  reset_kappa_image_target_sptr(input_image);
  info("Computing the spatially variant penalty strength using approximate hessian, this may take a while.");
  // Setup image
  auto ones_image_sptr = input_image->get_empty_copy();
  ones_image_sptr->fill(1.);

  // Approximate Hessian computation will error for a lot of priors so we ignore it!
  if (get_compute_with_penalty())
    info("Priors do not have an approximation of the Hessian. Ignoring the prior!");
  objective_function_sptr->add_multiplication_with_approximate_Hessian_without_penalty(*kappa_image_target_sptr,
                                                                                       *ones_image_sptr);

  // Kappa is defined as the sqrt of the output of add_multiplication_with_approximate_Hessian_without_penalty
  sqrt_image(*kappa_image_target_sptr);
}

template <typename TargetT>
void
KappaComputation<TargetT>::
sqrt_image(TargetT& image)
{
  // Square root the output
  typename TargetT::const_full_iterator output_iter = image.begin_all_const();
  const typename TargetT::const_full_iterator end_prior_output_iter = image.end_all_const();
  typename TargetT::full_iterator tmp_iter = image.begin_all();
  while (output_iter!=end_prior_output_iter)
  {
    *tmp_iter = sqrt(*output_iter);
    ++tmp_iter; ++output_iter;
  }
}

template class KappaComputation<DiscretisedDensity<3,float> >;
//template class KappaComputation<ParametricVoxelsOnCartesianGrid >;

END_NAMESPACE_STIR
