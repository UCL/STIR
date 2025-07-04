/*
    Copyright (C) 2003 - 2011-06-29, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2012, Kris Thielemans
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup GeneralisedObjectiveFunction
  \brief Implementation of class stir::PoissonLogLikelihoodWithLinearModelForMean

  \author Kris Thielemans
  \author Sanida Mustafovic
*/
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMean.h"
#include "stir/DiscretisedDensity.h"
#include "stir/is_null_ptr.h"
#include "stir/IO/write_to_file.h"
#include "stir/IO/read_from_file.h"
#include "stir/Succeeded.h"
#include "stir/CPUTimer.h"
#include <algorithm>
#include <exception>
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/modelling/KineticParameters.h"
#include "stir/info.h"
#include "stir/error.h"
#include "stir/format.h"
#include "boost/lexical_cast.hpp"
#include "boost/format.hpp"

using std::string;

START_NAMESPACE_STIR

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::set_defaults()
{
  base_type::set_defaults();

  this->sensitivity_filename = "";
  this->subsensitivity_filenames = "";
  this->recompute_sensitivity = false;
  this->use_subset_sensitivities = true;
  this->subsensitivity_sptrs.resize(0);
}

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::initialise_keymap()
{
  base_type::initialise_keymap();

  this->parser.add_key("sensitivity filename", &this->sensitivity_filename);
  this->parser.add_key("subset sensitivity filenames", &this->subsensitivity_filenames);
  this->parser.add_key("recompute sensitivity", &this->recompute_sensitivity);
  this->parser.add_key("use_subset_sensitivities", &this->use_subset_sensitivities);
}

template <typename TargetT>
bool
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::post_processing()
{
  if (base_type::post_processing() == true)
    return true;

  return false;
}

template <typename TargetT>
std::string
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::get_sensitivity_filename() const
{
  return this->sensitivity_filename;
}

template <typename TargetT>
std::string
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::get_subsensitivity_filenames() const
{
  return this->subsensitivity_filenames;
}

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::set_sensitivity_filename(const std::string& filename)
{
  this->already_set_up = false;
  this->sensitivity_filename = filename;
}

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::set_subsensitivity_filenames(const std::string& filenames)
{
  this->already_set_up = false;
  this->subsensitivity_filenames = filenames;
  try
    {
      if (this->subsensitivity_filenames.find("%"))
        {
          warning("The subsensitivity_filenames pattern is using the boost::format convention ('\%d')."
                  "It is recommended to use fmt::format/std::format style formatting ('{}').");
          const std::string test_sensitivity_filename = boost::str(boost::format(this->subsensitivity_filenames) % 0);
        }
      else
        {
          const std::string test_sensitivity_filename = format(this->subsensitivity_filenames.c_str(), 0);
        }
    }
  catch (std::exception& e)
    {
      error(format("argument {} to set_subsensitivity_filenames is invalid (see fmt::format documentation)\n. Error message: {}",
                   filenames.c_str(),
                   e.what()));
    }
}

template <typename TargetT>
shared_ptr<TargetT>
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::get_subset_sensitivity_sptr(const int subset_num) const
{
  return this->subsensitivity_sptrs[subset_num];
}

template <typename TargetT>
const TargetT&
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::get_subset_sensitivity(const int subset_num) const
{
  return *get_subset_sensitivity_sptr(subset_num);
}

template <typename TargetT>
const TargetT&
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::get_sensitivity() const
{
  return *this->sensitivity_sptr;
}

template <typename TargetT>
bool
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::get_recompute_sensitivity() const
{
  return this->recompute_sensitivity;
}

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::set_recompute_sensitivity(const bool arg)
{
  this->recompute_sensitivity = arg;
}

template <typename TargetT>
bool
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::get_use_subset_sensitivities() const
{
  return this->use_subset_sensitivities;
}

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::set_use_subset_sensitivities(const bool arg)
{
  this->already_set_up = this->already_set_up && (this->use_subset_sensitivities == arg);
  this->use_subset_sensitivities = arg;
}

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::set_subset_sensitivity_sptr(const shared_ptr<TargetT>& arg,
                                                                                 const int subset_num)
{
  this->already_set_up = false;
  this->subsensitivity_sptrs[subset_num] = arg;
}

template <typename TargetT>
Succeeded
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::set_up(shared_ptr<TargetT> const& target_sptr)
{
  if (base_type::set_up(target_sptr) != Succeeded::yes)
    return Succeeded::no;

  this->subsensitivity_sptrs.resize(this->num_subsets);

  if (!this->recompute_sensitivity)
    {
      if (is_null_ptr(this->subsensitivity_sptrs[0])
          && ((this->get_use_subset_sensitivities() && this->subsensitivity_filenames == "")
              || (!this->get_use_subset_sensitivities() && this->sensitivity_filename == "")))
        {
          info("(subset)sensitivity filename(s) not set so I will compute the (subset)sensitivities", 2);
          this->recompute_sensitivity = true;
          // initialisation of pointers will be done below
        }
      else if (this->sensitivity_filename == "1")
        {
          if (this->get_use_subset_sensitivities())
            {
              error("PoissonLogLikelihoodWithLinearModelForMean limitation:\n"
                    "currently cannot use subset_sensitivities if sensitivity is forced to 1");
              return Succeeded::no;
            }
          this->sensitivity_sptr.reset(target_sptr->get_empty_copy());
          std::fill(this->sensitivity_sptr->begin_all(), this->sensitivity_sptr->end_all(), 1);
        }
      else
        {
          // read from file
          try
            {
              if (this->get_use_subset_sensitivities())
                {
                  if (this->subsensitivity_filenames.empty())
                    {
                      error("'subset sensitivity filenames' is empty. You need to set this before using it.");
                      return Succeeded::no;
                    }
                  // read subsensitivies
                  for (int subset = 0; subset < this->get_num_subsets(); ++subset)
                    {
                      std::string current_sensitivity_filename;
                      try
                        {
                          if (this->subsensitivity_filenames.find("%"))
                            {
                              warning("The subsensitivity_filenames pattern is using the boost::format convention ('\%d')."
                                      "It is recommended to use fmt::format/std::format style formatting ('{}').");
                              current_sensitivity_filename = boost::str(boost::format(this->subsensitivity_filenames) % subset);
                            }
                          else
                            {
                              current_sensitivity_filename = runtime_format(this->subsensitivity_filenames.c_str(), subset);
                            }
                        }
                      catch (std::exception& e)
                        {
                          error(format("Error using 'subset sensitivity filenames' pattern (which is set to '{}'). "
                                       "Check syntax for fmt::format. Error is:\n{}",
                                       this->subsensitivity_filenames,
                                       e.what()));
                          return Succeeded::no;
                        }
                      info(format("Reading sensitivity from '{}'", current_sensitivity_filename));

                      this->subsensitivity_sptrs[subset] = read_from_file<TargetT>(current_sensitivity_filename);
                      string explanation;
                      if (!target_sptr->has_same_characteristics(*this->subsensitivity_sptrs[subset], explanation))
                        {
                          error("sensitivity and target should have the same characteristics.\n%s", explanation.c_str());
                          return Succeeded::no;
                        }
                    }
                }
              else
                {
                  if (this->sensitivity_filename.empty())
                    {
                      error("'sensitivity filename' is empty. You need to set this before using it.");
                      return Succeeded::no;
                    }
                  // reading single sensitivity
                  const std::string current_sensitivity_filename = this->sensitivity_filename;
                  info(format("Reading sensitivity from '{}'", current_sensitivity_filename));

                  this->sensitivity_sptr = read_from_file<TargetT>(current_sensitivity_filename);
                  string explanation;
                  if (!target_sptr->has_same_characteristics(*this->sensitivity_sptr, explanation))
                    {
                      error("sensitivity and target should have the same characteristics.\n%s", explanation.c_str());
                      return Succeeded::no;
                    }
                }
            }
          catch (std::exception& e)
            {
              error("Error reading sensitivity from file:\n%s", e.what());
              return Succeeded::no;
            }
          // compute total from subsensitivity or vice versa
          this->set_total_or_subset_sensitivities();
        }
    } // end of !recompute_sensitivity case

  if (this->set_up_before_sensitivity(target_sptr) == Succeeded::no)
    {
      return Succeeded::no;
    }

  if (!this->subsets_are_approximately_balanced() && !this->get_use_subset_sensitivities())
    {
      error("Number of subsets %d is such that subsets will be very unbalanced.\n"
            "You need to set 'use_subset_sensitivities' to true to handle this.",
            this->num_subsets);
      return Succeeded::no;
    }

  if (this->recompute_sensitivity)
    {
      info("Computing sensitivity");
      CPUTimer sens_timer;
      sens_timer.start();
      // preallocate one such that compute_sensitivities knows the size
      this->subsensitivity_sptrs[0].reset(target_sptr->get_empty_copy());
      this->compute_sensitivities();
      sens_timer.stop();
      info("Done computing sensitivity");
      info("This took " + boost::lexical_cast<std::string>(sens_timer.value()) + " seconds CPU time.");

      // write to file
      try
        {
          if (this->get_use_subset_sensitivities())
            {
              if (this->subsensitivity_filenames.size() != 0)
                {
                  for (int subset = 0; subset < this->get_num_subsets(); ++subset)
                    {
                      const std::string current_sensitivity_filename
                          = runtime_format(this->subsensitivity_filenames.c_str(), subset);
                      info(format("Writing sensitivity to '{}'", current_sensitivity_filename));
                      write_to_file(current_sensitivity_filename, this->get_subset_sensitivity(subset));
                    }
                }
            }
          else
            {
              if (this->sensitivity_filename.size() != 0)
                {
                  const std::string current_sensitivity_filename = this->sensitivity_filename;
                  info(format("Writing sensitivity to '{}'", current_sensitivity_filename));
                  write_to_file(current_sensitivity_filename, this->get_sensitivity());
                }
            }
        }
      catch (std::exception& e)
        {
          error("Error writing sensitivity to file:\n%s", e.what());
          return Succeeded::no;
        }
    }
  this->already_set_up = true;
  return Succeeded::yes;
}

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::compute_sub_gradient_without_penalty(TargetT& gradient,
                                                                                          const TargetT& current_estimate,
                                                                                          const int subset_num)
{
  if (!this->already_set_up)
    error("Need to call set_up() for objective function first");
  this->actual_compute_subset_gradient_without_penalty(gradient, current_estimate, subset_num, false);
}

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::compute_sub_gradient_without_penalty_plus_sensitivity(
    TargetT& gradient, const TargetT& current_estimate, const int subset_num)
{
  if (!this->already_set_up)
    error("Need to call set_up() for objective function first");
  actual_compute_subset_gradient_without_penalty(gradient, current_estimate, subset_num, true);
}

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::compute_sensitivities()
{
  // check subset balancing
  if (this->use_subset_sensitivities == false)
    {
      std::string warning_message = "PoissonLogLikelihoodWithLinearModelForMean:\n";
      if (!this->subsets_are_approximately_balanced(warning_message))
        {
          error("%s\n . you need to set use_subset_sensitivities to true", warning_message.c_str());
        }
    } // end check balancing

  // compute subset sensitivities
  for (int subset_num = 0; subset_num < this->num_subsets; ++subset_num)
    {
      if (subset_num == 0)
        {
          std::fill(this->subsensitivity_sptrs[subset_num]->begin_all(), this->subsensitivity_sptrs[subset_num]->end_all(), 0);
        }
      else
        {
          if (this->get_use_subset_sensitivities())
            {
              this->subsensitivity_sptrs[subset_num].reset(this->subsensitivity_sptrs[0]->get_empty_copy());
            }
          else
            {
              // copy subsensitivity[0] pointer to current subset.
              // do this as pointer such that we don't use more memory.
              // This also means that we just accumulate in subsensitivity[0] for the moment.
              // we will correct that below.
              this->subsensitivity_sptrs[subset_num] = this->subsensitivity_sptrs[0];
            }
        }
      this->add_subset_sensitivity(*this->get_subset_sensitivity_sptr(subset_num), subset_num);
    }
  if (!this->get_use_subset_sensitivities())
    {
      // copy full sensitivity (currently stored in subsensitivity[0])
      this->sensitivity_sptr = this->subsensitivity_sptrs[0];
      this->subsensitivity_sptrs[0].reset();
    }
  // compute total from subsensitivity or vice versa
  this->set_total_or_subset_sensitivities();
}

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::set_total_or_subset_sensitivities()
{
  if (this->get_use_subset_sensitivities())
    {
      // add subset sensitivities, just in case we need the total somewhere
      this->sensitivity_sptr.reset(this->subsensitivity_sptrs[0]->clone());
      for (int subset_num = 1; subset_num < this->num_subsets; ++subset_num)
        {
          typename TargetT::full_iterator sens_iter = this->sensitivity_sptr->begin_all();
          typename TargetT::full_iterator subsens_iter = this->subsensitivity_sptrs[subset_num]->begin_all();
          while (sens_iter != this->sensitivity_sptr->end_all())
            {
              *sens_iter += *subsens_iter;
              ++sens_iter;
              ++subsens_iter;
            }
        }
    }
  else
    {
      // copy full sensitivity
      this->subsensitivity_sptrs[0].reset(this->sensitivity_sptr->clone());
      // divide subsensitivity[0] by num_subsets
      for (typename TargetT::full_iterator subsens_iter = this->subsensitivity_sptrs[0]->begin_all();
           subsens_iter != this->subsensitivity_sptrs[0]->end_all();
           ++subsens_iter)
        {
          *subsens_iter /= this->num_subsets;
        }
      // set all other pointers the same
      for (int subset_num = 1; subset_num < this->num_subsets; ++subset_num)
        this->subsensitivity_sptrs[subset_num] = this->subsensitivity_sptrs[0];
    }
}

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::fill_nonidentifiable_target_parameters(TargetT& target,
                                                                                            const float value) const
{
  typename TargetT::full_iterator target_iter = target.begin_all();
  typename TargetT::full_iterator target_end_iter = target.end_all();
  typename TargetT::const_full_iterator sens_iter = this->get_sensitivity().begin_all_const();

  for (; target_iter != target_end_iter; ++target_iter, ++sens_iter)
    {
      if (*sens_iter == 0)
        *target_iter = value;
    }
}

#ifdef _MSC_VER
// prevent warning message on instantiation of abstract class
#  pragma warning(disable : 4661)
#endif

template class PoissonLogLikelihoodWithLinearModelForMean<DiscretisedDensity<3, float>>;
template class PoissonLogLikelihoodWithLinearModelForMean<ParametricVoxelsOnCartesianGrid>;

END_NAMESPACE_STIR
