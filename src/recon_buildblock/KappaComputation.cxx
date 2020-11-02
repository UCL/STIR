//
// Created by Robert Twyman on 30/10/2020.
//

#include "stir/recon_buildblock/KappaComputation.h"
#include "stir/info.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/IO/read_from_file.h"
#include "stir/is_null_ptr.h"
#include "stir/recon_buildblock/GeneralisedObjectiveFunction.h"
#include "stir/recon_buildblock/GeneralisedPrior.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h"


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
}

template <typename TargetT>
void
KappaComputation<TargetT>::initialise_keymap()
{
  parser.add_start_key("Kappa Computation Parameters");
  parser.add_key("kappa filename", &kappa_filename);
  parser.add_key("current image estimate", &current_image_estimate_filename);
  parser.add_key("template image", &template_image_filename);
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

  if (current_image_estimate_filename.empty() && template_image_filename.empty())
  {
    error("Requires either current_image_estimate_filename or template_image_filename");
    return true;
  }
  return false;
}

template <typename TargetT>
PoissonLogLikelihoodWithLinearModelForMean<TargetT > const&
KappaComputation<TargetT>::
get_objective_function()
{
  return *objective_function_sptr;
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
get_current_image_estimate()
{
  return current_image_estimate_sptr;
}

template <typename TargetT>
void
KappaComputation<TargetT>::
set_current_image_estimate(shared_ptr <TargetT > const& image)
{
  current_image_estimate_sptr = image;
}

template <typename TargetT>
shared_ptr <TargetT>
KappaComputation<TargetT>::
get_template_image()
{
  return template_image_sptr;
}

template <typename TargetT>
void
KappaComputation<TargetT>::
set_template_image(shared_ptr <TargetT > const& image)
{
  template_image_sptr = image;
}


template <typename TargetT>
void
KappaComputation<TargetT>::
process_data()
{
  // Unfortunately we have to setup. Therefore we will cheat with regards to sensitivities that are unused in the kappa
  // computation.
  // To prevent sensitivity from being computed (which takes a lot of computation), the sensitivity filename is set to
  // a filename known to exist, one of the file inputs, and re-computation is turned off.
  objective_function_sptr->set_recompute_sensitivity(false);
  objective_function_sptr->set_use_subset_sensitivities(false);

  if (!current_image_estimate_filename.empty())
  {
    current_image_estimate_sptr = read_from_file<TargetT>(current_image_estimate_filename);
    objective_function_sptr->set_sensitivity_filename(current_image_estimate_filename);
    objective_function_sptr->set_up(current_image_estimate_sptr);
    compute_kappa_at_current_image_estimate();
  }

  else if (!template_image_filename.empty())
  {
    template_image_sptr = read_from_file<TargetT>(template_image_filename);
    objective_function_sptr->set_sensitivity_filename(template_image_filename);
    objective_function_sptr->set_up(template_image_sptr);
    compute_kappa_with_approximate();
  }

  else
    error("process_data: Either both current_image_estimate_filename and template_image_filename are empty.");

  info("Spatially variant penalty strength (Kappa) has been computed and saved as " + kappa_filename + ".");
}


template <typename TargetT>
void
KappaComputation<TargetT>::
compute_kappa_at_current_image_estimate()
{
  info("Computing the spatially variant penalty strength at the current image estimate.");
  // Setup image
  auto output_image_sptr = current_image_estimate_sptr->get_empty_copy();
  output_image_sptr->fill(0);
  auto ones_image_sptr = current_image_estimate_sptr->get_empty_copy();
  ones_image_sptr->fill(1);

  objective_function_sptr->accumulate_Hessian_times_input(*output_image_sptr, *current_image_estimate_sptr, *ones_image_sptr);

  // Kappa is defined as the sqrt of the output of accumulate_Hessian_times_input
  sqrt_image(*output_image_sptr);

  // Save the output
  output_file_format_sptr->write_to_file(kappa_filename, *output_image_sptr);
}

template <typename TargetT>
void
KappaComputation<TargetT>::
compute_kappa_with_approximate()
{

  info("Computing the spatially variant penalty strength using approximate hessian.");
  // Setup image
  auto output_image_sptr = template_image_sptr->get_empty_copy();
  output_image_sptr->fill(0.);
  auto ones_image_sptr = template_image_sptr->get_empty_copy();
  ones_image_sptr->fill(1.);

  // Approximate Hessian computation will error for a lot of priors so we ignore it!
  info("Priors do not have an approximation of the Hessian. Therefore we will ignore the prior.");
  objective_function_sptr->add_multiplication_with_approximate_Hessian_without_penalty(*output_image_sptr, *ones_image_sptr);

  // Kappa is defined as the sqrt of the output of add_multiplication_with_approximate_Hessian_without_penalty
  sqrt_image(*output_image_sptr);

  // Save the output
  output_file_format_sptr->write_to_file(kappa_filename, *output_image_sptr);
}

template <typename TargetT>
void
KappaComputation<TargetT>::
sqrt_image(TargetT& output_image_sptr)
{
  // Square root the output
  typename TargetT::const_full_iterator output_iter = output_image_sptr.begin_all_const();
  const typename TargetT::const_full_iterator end_prior_output_iter = output_image_sptr.end_all_const();
  typename TargetT::full_iterator tmp_iter = output_image_sptr.begin_all();
  while (output_iter!=end_prior_output_iter)
  {
    *tmp_iter = sqrt(*output_iter);
    ++tmp_iter; ++output_iter;
  }
}

template class KappaComputation<DiscretisedDensity<3,float> >;
//template class KappaComputation<ParametricVoxelsOnCartesianGrid >;

END_NAMESPACE_STIR
