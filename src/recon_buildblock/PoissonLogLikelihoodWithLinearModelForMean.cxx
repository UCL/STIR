//
// $Id$
//
/*
    Copyright (C) 2003 - 2011-06-29, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - $Date$, Kris Thielemans
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
*/
/*!
  \file
  \ingroup GeneralisedObjectiveFunction
  \brief Implementation of class stir::PoissonLogLikelihoodWithLinearModelForMean

  \author Kris Thielemans
  \author Sanida Mustafovic

  $Date$
  $Revision$
*/
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMean.h"
#include "stir/DiscretisedDensity.h"
#include "stir/is_null_ptr.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/IO/read_from_file.h"
#include "stir/Succeeded.h"
#include <algorithm>
#include <exception>
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/modelling/KineticParameters.h"
#include "stir/info.h"
#include "boost/format.hpp"

START_NAMESPACE_STIR

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::
set_defaults()
{
  base_type::set_defaults();

  this->sensitivity_filename = "";  
  this->subsensitivity_filenames = "";  
  this->recompute_sensitivity = false;
  this->use_subset_sensitivities = false;
  this->subsensitivity_sptrs.resize(0);
}

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::
initialise_keymap()
{
  base_type::initialise_keymap();

  this->parser.add_key("sensitivity filename", &this->sensitivity_filename);
  this->parser.add_key("subset sensitivity filenames", &this->subsensitivity_filenames);
  this->parser.add_key("recompute sensitivity", &this->recompute_sensitivity);
  this->parser.add_key("use_subset_sensitivities", &this->use_subset_sensitivities);

}

template<typename TargetT>
bool
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::
post_processing()
{
  if (base_type::post_processing() == true)
    return true;

  return false;
}

template<typename TargetT>
std::string
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::
get_sensitivity_filename() const
{
  return this->sensitivity_filename;
}

template<typename TargetT>
std::string
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::
get_subsensitivity_filenames() const
{
  return this->subsensitivity_filenames;
}

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::
set_sensitivity_filename(const std::string& filename)
{
  this->sensitivity_filename = filename;
}

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::
set_subsensitivity_filenames(const std::string& filenames)
{
  this->subsensitivity_filenames = filenames;
  try
    {
      const std::string test_sensitivity_filename =
	boost::str(boost::format(this->subsensitivity_filenames) % 0);
    }
  catch (std::exception& e)
    {
      error("argument %s to set_subsensitivity_filenames is invalid (see boost::format documentation)\n. Error message: %s", filenames.c_str(), e.what());
    }

}


template<typename TargetT>
shared_ptr<TargetT> 
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::
get_subset_sensitivity_sptr(const int subset_num) const
{
  return this->subsensitivity_sptrs[subset_num];
}

template<typename TargetT>
const TargetT&
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::
get_subset_sensitivity(const int subset_num) const
{
  return *get_subset_sensitivity_sptr(subset_num);
}

template<typename TargetT>
const TargetT&
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::
get_sensitivity() const
{
  return *this->sensitivity_sptr;
}

template<typename TargetT>
bool
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::
get_recompute_sensitivity() const
{
  return this->recompute_sensitivity;
}

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::
set_recompute_sensitivity(const bool arg)
{
  this->recompute_sensitivity = arg;

}

template<typename TargetT>
bool
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::
get_use_subset_sensitivities() const
{
  return this->use_subset_sensitivities;
}

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::
set_use_subset_sensitivities(const bool arg)
{
  this->use_subset_sensitivities = arg;
}

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::
set_subset_sensitivity_sptr(const shared_ptr<TargetT>& arg, const int subset_num)
{
  this->subsensitivity_sptrs[subset_num] = arg;
}

template<typename TargetT>
Succeeded 
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::
set_up(shared_ptr<TargetT> const& target_sptr)
{
  if (base_type::set_up(target_sptr) != Succeeded::yes)
    return Succeeded::no;

  this->subsensitivity_sptrs.resize(this->num_subsets);

  if(!this->recompute_sensitivity)
    {      
      if(is_null_ptr(this->subsensitivity_sptrs[0]) &&
         ((this->get_use_subset_sensitivities() && this->subsensitivity_filenames=="") ||
          (!this->get_use_subset_sensitivities() && this->sensitivity_filename=="")))
        {
          warning("recompute_sensitivity is set to false, but sensitivity pointer is empty "
                  "and (sub)sensitivity filename is not set. I will compute the sensitivity anyway.");
          this->recompute_sensitivity = true;
          // initialisation of pointers will be done below
        }
      else if(this->sensitivity_filename=="1")
        {
          if (this->get_use_subset_sensitivities())
            {
              warning("PoissonLogLikelihoodWithLinearModelForMean limitation:\n"
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
                  // read subsensitivies
                  for (int subset=0; subset<this->get_num_subsets(); ++subset)
                    {
                      const std::string current_sensitivity_filename =
                        boost::str(boost::format(this->subsensitivity_filenames) % subset);
                      info(boost::format("Reading sensitivity from '%1%'") % current_sensitivity_filename);

                      this->subsensitivity_sptrs[subset] = 
                        read_from_file<TargetT>(current_sensitivity_filename);   
                      string explanation;
                      if (!target_sptr->has_same_characteristics(*this->subsensitivity_sptrs[subset], 
                                                                 explanation))
                        {
                          warning("sensitivity and target should have the same characteristics.\n%s",
                                  explanation.c_str());
                          return Succeeded::no;
                        }
                    }
                }
              else
                {
                  // reading single sensitivity
                  const std::string current_sensitivity_filename =
                    this->sensitivity_filename;
                  info(boost::format("Reading sensitivity from '%1%'") % current_sensitivity_filename);

                  this->sensitivity_sptr = read_from_file<TargetT>(current_sensitivity_filename);
                  string explanation;
                  if (!target_sptr->has_same_characteristics(*this->sensitivity_sptr, 
                                                             explanation))
                    {
                      warning("sensitivity and target should have the same characteristics.\n%s",
                              explanation.c_str());
                      return Succeeded::no;
                    }
                }
            }
          catch (std::exception& e)
            {
              warning("Error reading sensitivity from file:\n%s", e.what());
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

  if(!this->subsets_are_approximately_balanced() && !this->get_use_subset_sensitivities())
    {
      warning("Number of subsets %d is such that subsets will be very unbalanced.\n"
              "You need to set 'use_subset_sensitivities' to true to handle this.",
              this->num_subsets);
      return Succeeded::no;
    }

  if(this->recompute_sensitivity)
    {
      info("Computing sensitivity");      
      // preallocate one such that compute_sensitivities knows the size
      this->subsensitivity_sptrs[0].reset(target_sptr->get_empty_copy());
      this->compute_sensitivities();
      info("Done computing sensitivity");

      // write to file
      try
        {
          if (this->get_use_subset_sensitivities())
            {
              if (this->subsensitivity_filenames.size()!=0)
                {
                  for (int subset=0; subset<this->get_num_subsets(); ++subset)
                    {
                      const std::string current_sensitivity_filename =
                        boost::str(boost::format(this->subsensitivity_filenames) % subset);
                      info(boost::format("Writing sensitivity to '%1%'") % current_sensitivity_filename);
                      OutputFileFormat<TargetT>::default_sptr()->
                        write_to_file(current_sensitivity_filename,
                                      this->get_subset_sensitivity(subset));
                    }
                }
            }
          else
            {
              if (this->sensitivity_filename.size()!=0)
                {            
                  const std::string current_sensitivity_filename =
                    this->sensitivity_filename;
                  info(boost::format("Writing sensitivity to '%1%'") % current_sensitivity_filename);
                  OutputFileFormat<TargetT>::default_sptr()->
                    write_to_file(current_sensitivity_filename,
                                  this->get_sensitivity());
                }
            }
        }
      catch (std::exception& e)
        {
          warning("Error writing sensitivity to file:\n%s", e.what());
          return Succeeded::no;
        }
    }
      
  return Succeeded::yes;
}

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::
compute_sub_gradient_without_penalty(TargetT& gradient, 
                                     const TargetT &current_estimate, 
                                     const int subset_num)
{
  this->
    compute_sub_gradient_without_penalty_plus_sensitivity(gradient, 
                                                          current_estimate,
                                                          subset_num);
  // compute gradient -= sub_sensitivity
  {
    typename TargetT::full_iterator gradient_iter =
      gradient.begin_all();
    const typename TargetT::full_iterator gradient_end = 
      gradient.end_all();
    typename TargetT::const_full_iterator sensitivity_iter =
      this->get_subset_sensitivity(subset_num).begin_all_const();
    while (gradient_iter != gradient_end)
      {
        *gradient_iter -= (*sensitivity_iter);
        ++gradient_iter; ++sensitivity_iter;
      }
  }
}


template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::
compute_sensitivities()
{
  // check subset balancing
  if (this->use_subset_sensitivities == false)
  {
    std::string warning_message = "PoissonLogLikelihoodWithLinearModelForMean:\n";
    if (!this->subsets_are_approximately_balanced(warning_message))
      {
        error("%s\n . you need to set use_subset_sensitivities to true",
                warning_message.c_str());
      }
  } // end check balancing

  // compute subset sensitivities
  for (int subset_num=0; subset_num<this->num_subsets; ++subset_num)
    {
      if (subset_num == 0)
        {
          std::fill(this->subsensitivity_sptrs[subset_num]->begin_all(), 
                    this->subsensitivity_sptrs[subset_num]->end_all(), 
                    0);
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

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::
set_total_or_subset_sensitivities()
{
  if (this->get_use_subset_sensitivities())
    {
      // add subset sensitivities, just in case we need the total somewhere
      this->sensitivity_sptr.reset(this->subsensitivity_sptrs[0]->clone());
      for (int subset_num=1; subset_num<this->num_subsets; ++subset_num)
        {
          typename TargetT::full_iterator sens_iter = this->sensitivity_sptr->begin_all();
          typename TargetT::full_iterator subsens_iter = this->subsensitivity_sptrs[subset_num]->begin_all();
          while (sens_iter != this->sensitivity_sptr->end_all())
            {
              *sens_iter += *subsens_iter;
              ++sens_iter; ++ subsens_iter;
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
      for (int subset_num=1; subset_num<this->num_subsets; ++subset_num)
        this->subsensitivity_sptrs[subset_num] = this->subsensitivity_sptrs[0];
    }

}


template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::
fill_nonidentifiable_target_parameters(TargetT& target, const float value) const
{
  typename TargetT::full_iterator target_iter = target.begin_all();
  typename TargetT::full_iterator target_end_iter = target.end_all();
  typename TargetT::const_full_iterator sens_iter = 
    this->get_sensitivity().begin_all_const();
  
  for (;
       target_iter != target_end_iter;
       ++target_iter, ++sens_iter)
    {
      if (*sens_iter == 0)
        *target_iter = value;
    }
}

#  ifdef _MSC_VER
// prevent warning message on instantiation of abstract class 
#  pragma warning(disable:4661)
#  endif

template class PoissonLogLikelihoodWithLinearModelForMean<DiscretisedDensity<3,float> >;
template class PoissonLogLikelihoodWithLinearModelForMean<ParametricVoxelsOnCartesianGrid >; 

END_NAMESPACE_STIR


