//
// $Id$
//
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
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
#include "stir/Succeeded.h"
#include <algorithm>
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/modelling/KineticParameters.h"

START_NAMESPACE_STIR

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::
set_defaults()
{
  base_type::set_defaults();

  this->sensitivity_filename = "";  
  this->recompute_sensitivity = false;
  this->use_subset_sensitivities = false;
  this->sensitivity_sptrs.resize(1);
  this->sensitivity_sptrs[0] = 0;

}

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::
initialise_keymap()
{
  base_type::initialise_keymap();

  this->parser.add_key("sensitivity filename", &this->sensitivity_filename);
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
shared_ptr<TargetT> 
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::
get_subset_sensitivity_sptr(const int subset_num) const
{
  const int actual_subset_num =
    this->get_use_subset_sensitivities()
    ? subset_num
    : 0;
  return this->sensitivity_sptrs[actual_subset_num];
}

template<typename TargetT>
const TargetT&
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::
get_sensitivity(const int subset_num) const
{
  return *get_subset_sensitivity_sptr(subset_num);
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
set_sensitivity_sptr(const shared_ptr<TargetT>& arg, const int subset_num)
{
  this->sensitivity_sptrs[subset_num] = arg;
}

template<typename TargetT>
Succeeded 
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::
set_up(shared_ptr<TargetT> const& target_sptr)
{
  if (base_type::set_up(target_sptr) != Succeeded::yes)
    return Succeeded::no;

#if 0
  // TODO cannot call this yet as projectors are not yet set-up
  // check subset balancing
  if (this->use_subset_sensitivities == false)
  {
    std::string warning_message = "PoissonLogLikelihoodWithLinearModelForMean:\n";
    if (!this->subsets_are_approximately_balanced(warning_message))
      {
	warning("%s\n . you need to set use_subset_sensitivities to true",
		warning_message.c_str());
	return Succeeded::no;
      }
  } // end check balancing
#endif
  if(!this->recompute_sensitivity)
    {
      if (this->get_use_subset_sensitivities())
	{
	  warning("PoissonLogLikelihoodWithLinearModelForMean limitation:\n"
		  "currently can only use subset_sensitivities if recompute_sensitivity==true");
	  return Succeeded::no;
	}
      
      if(this->sensitivity_filename=="")
	{
	  if (is_null_ptr(this->sensitivity_sptrs[0]))
	    {
	      warning("recompute_sensitivity is set to false, but sensitivity pointer is empty "
		      "and sensitivity filename is not set. I will compute the sensitivity anyway.");
	      this->recompute_sensitivity = true;
	      // initialisation of pointers will be done below
	    }
	}
      else if(this->sensitivity_filename=="1")
	{
	  this->sensitivity_sptrs[0]=target_sptr->get_empty_copy();
	  std::fill(this->sensitivity_sptrs[0]->begin_all(), this->sensitivity_sptrs[0]->end_all(), 1);  
	}
      else
	{       
	  
	  this->sensitivity_sptrs[0] = 
	    TargetT::read_from_file(this->sensitivity_filename);   
	  string explanation;
	  if (!target_sptr->has_same_characteristics(*this->sensitivity_sptrs[0], 
						     explanation))
	    {
	      warning("sensitivity and target should have the same characteristics.\n%s",
		      explanation.c_str());
	      return Succeeded::no;
	    }
	}
    } // end of !recompute_sensitivity case

  // handle recompute_sensitivity==true case
  // note: repeat if (as opposed to using "else") such that we get here if 
  // recompute_sensitivity was set above
  if(this->recompute_sensitivity)
    {
      if (this->use_subset_sensitivities == false)
	this->sensitivity_sptrs.resize(1);
      else
	this->sensitivity_sptrs.resize(this->num_subsets);

      for (int subset_num=0; subset_num < static_cast<int>(this->sensitivity_sptrs.size()); ++ subset_num)
	this->sensitivity_sptrs[subset_num]=target_sptr->get_empty_copy();

      // note: at this point, the projectors are not yet set-up, so we cannot
      // call compute_sensitivities here.
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
      this->get_sensitivity(subset_num).begin_all_const();
    while (gradient_iter != gradient_end)
      {
	*gradient_iter -= (*sensitivity_iter)/
	  (this->get_use_subset_sensitivities() == false ? this->num_subsets : 1);
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

  for (int subset_num=0; subset_num<this->num_subsets; ++subset_num)
    {
      if (this->get_use_subset_sensitivities() || subset_num == 0)
	{
	  std::fill(this->sensitivity_sptrs[subset_num]->begin_all(), 
		    this->sensitivity_sptrs[subset_num]->end_all(), 
		    0);
	}
      this->add_subset_sensitivity(*this->get_subset_sensitivity_sptr(subset_num), subset_num);
    }
  // TODO (but needs change in various bits and pieces)
  // if (!this->get_use_subset_sensitivities()) *sensitivity_sptr[0]/=num_subsets
}


template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMean<TargetT>::
fill_nonidentifiable_target_parameters(TargetT& target, const float value) const
{
  typename TargetT::full_iterator target_iter = target.begin_all();
  typename TargetT::full_iterator target_end_iter = target.end_all();
  // TODO really should use total sensitivity, not subset
  typename TargetT::const_full_iterator sens_iter = 
    this->get_sensitivity(0).begin_all_const();
  
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


