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
  \brief Declaration of class stir::GeneralisedObjectiveFunction

  \author Kris Thielemans
  \author Sanida Mustafovic

  $Date$
  $Revision$
*/

#include "stir/recon_buildblock/GeneralisedObjectiveFunction.h"
#include "stir/DiscretisedDensity.h"
#include "stir/is_null_ptr.h"
#include "stir/Succeeded.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/modelling/KineticParameters.h"

START_NAMESPACE_STIR

template <typename TargetT>
void
GeneralisedObjectiveFunction<TargetT>::
set_defaults()
{
  this->prior_sptr = 0;
  // note: cannot use set_num_subsets(1) here, as other parameters (such as projectors) are not set-up yet.
  this->num_subsets = 1;
}

template <typename TargetT>
void
GeneralisedObjectiveFunction<TargetT>::
initialise_keymap()
{
  this->parser.add_parsing_key("prior type", &prior_sptr);
}

template <typename TargetT>
GeneralisedObjectiveFunction<TargetT>::
~GeneralisedObjectiveFunction()
{}

template <typename TargetT>
Succeeded 
GeneralisedObjectiveFunction<TargetT>::
set_up(shared_ptr<TargetT> const& target_data_ptr)
{
  if (!is_null_ptr(this->prior_sptr) &&
      this->prior_sptr->set_up(target_data_ptr) == Succeeded::no)
    return Succeeded::no;

  if (this->num_subsets <= 0)
    {
      warning("Number of subsets %d should be larger than 0.",
	      this->num_subsets);
      return Succeeded::no;
    }

  return Succeeded::yes;  
}

template <typename TargetT>
GeneralisedPrior<TargetT> * const
GeneralisedObjectiveFunction<TargetT>::
get_prior_ptr() const
{ return this->prior_sptr.get(); }

template <typename TargetT>
void
GeneralisedObjectiveFunction<TargetT>::
set_prior_sptr(const shared_ptr<GeneralisedPrior<TargetT> >& arg)
{
  this->prior_sptr = arg;
}

template <typename TargetT>
bool
GeneralisedObjectiveFunction<TargetT>::
prior_is_zero() const
{
  return
    is_null_ptr(this->prior_sptr) ||
    this->prior_sptr->get_penalisation_factor() == 0;
}

template <typename TargetT>
double
GeneralisedObjectiveFunction<TargetT>::
compute_penalty(const TargetT& current_estimate)
{
  if (this->prior_is_zero())
    return 0.;
  else
    return this->prior_sptr->compute_value(current_estimate);
}

template <typename TargetT>
double
GeneralisedObjectiveFunction<TargetT>::
compute_penalty(const TargetT& current_estimate,
		const int subset_num)
{
  return this->compute_penalty(current_estimate)/this->get_num_subsets();
}

template <typename TargetT>
void 
GeneralisedObjectiveFunction<TargetT>::
compute_sub_gradient(TargetT& gradient, 
		     const TargetT &current_estimate, 
		     const int subset_num)
{
   assert(gradient.get_index_range() == current_estimate.get_index_range());  

   this->compute_sub_gradient_without_penalty(gradient, 
					      current_estimate, 
					      subset_num); 
   if (!this->prior_is_zero())
     {
       shared_ptr<TargetT>  prior_gradient_sptr =
	 gradient.get_empty_copy();
       this->prior_sptr->compute_gradient(*prior_gradient_sptr, current_estimate);

       // (*prior_gradient_sptr)/= num_subsets;
       // gradient -= *prior_gradient_sptr;
       typename TargetT::const_full_iterator prior_gradient_iter = prior_gradient_sptr->begin_all_const();
       const typename TargetT::const_full_iterator end_prior_gradient_iter = prior_gradient_sptr->end_all_const();
       typename TargetT::full_iterator gradient_iter = gradient.begin_all();
       while (prior_gradient_iter!=end_prior_gradient_iter)
	 {
	   *gradient_iter -= (*prior_gradient_iter)/this->get_num_subsets();
	   ++gradient_iter; ++prior_gradient_iter;
	 }
     }
}

template <typename TargetT>
int 
GeneralisedObjectiveFunction<TargetT>::
get_num_subsets() const
{
  return this->num_subsets;
}

template <typename TargetT>
double
GeneralisedObjectiveFunction<TargetT>::
compute_objective_function_without_penalty(const TargetT& current_estimate)
{
  double result = 0.;
  for (int subset_num=0; subset_num<this->get_num_subsets(); ++subset_num)
    result += 
      this->compute_objective_function_without_penalty(current_estimate, subset_num);
  return result;
}

template <typename TargetT>
double
GeneralisedObjectiveFunction<TargetT>::
compute_objective_function_without_penalty(const TargetT& current_estimate,
					   const int subset_num)
{
  return
    this->actual_compute_objective_function_without_penalty(current_estimate, subset_num);
}

template <typename TargetT>
double
GeneralisedObjectiveFunction<TargetT>::
compute_objective_function(const TargetT& current_estimate,
			   const int subset_num)
{
  return
    this->compute_objective_function_without_penalty(current_estimate, subset_num) -
    this->compute_penalty(current_estimate, subset_num);
}

template <typename TargetT>
double
GeneralisedObjectiveFunction<TargetT>::
compute_objective_function(const TargetT& current_estimate)
{
  return
    this->compute_objective_function_without_penalty(current_estimate) -
    this->compute_penalty(current_estimate);
}

/////////////////////// Hessian

template <typename TargetT>
Succeeded 
GeneralisedObjectiveFunction<TargetT>::
add_multiplication_with_approximate_sub_Hessian_without_penalty(TargetT& output,
								const TargetT& input,
								const int subset_num) const
{
  {
    string explanation;
    if (!output.has_same_characteristics(input,
					 explanation))
      {
	warning("GeneralisedObjectiveFunction:\n"
		"input and output for add_multiplication_with_approximate_sub_Hessian_without_penalty\n"
		"should have the same characteristics.\n%s",
		explanation.c_str());
	return Succeeded::no;
      }
  }
  return
   this->actual_add_multiplication_with_approximate_sub_Hessian_without_penalty(output,
										input,
										subset_num); 
}

template <typename TargetT>
Succeeded 
GeneralisedObjectiveFunction<TargetT>::
add_multiplication_with_approximate_sub_Hessian(TargetT& output,
						const TargetT& input,
						const int subset_num) const
{
  if (this->add_multiplication_with_approximate_sub_Hessian_without_penalty(output, input, subset_num) ==
      Succeeded::no)
    return Succeeded::no;

  if (!this->prior_is_zero())
    {
      // TODO used boost:scoped_ptr
      shared_ptr<TargetT>  prior_output_sptr =
	 output.get_empty_copy();
      if (this->prior_sptr->add_multiplication_with_approximate_Hessian(*prior_output_sptr, output) ==
	  Succeeded::no)
	return Succeeded::no;


       // (*prior_output_sptr)/= num_subsets;
       // output -= *prior_output_sptr;
       typename TargetT::const_full_iterator prior_output_iter = prior_output_sptr->begin_all_const();
       const typename TargetT::const_full_iterator end_prior_output_iter = prior_output_sptr->end_all_const();
       typename TargetT::full_iterator output_iter = output.begin_all();
       while (prior_output_iter!=end_prior_output_iter)
	 {
	   *output_iter -= (*prior_output_iter)/this->get_num_subsets();
	   ++output_iter; ++prior_output_iter;
	 }
     }

  return Succeeded::yes;

}


template <typename TargetT>
Succeeded 
GeneralisedObjectiveFunction<TargetT>::
add_multiplication_with_approximate_Hessian_without_penalty(TargetT& output,
							    const TargetT& input) const
{
  for (int subset_num=0; subset_num<this->get_num_subsets(); ++subset_num)
    {
      if (this->add_multiplication_with_approximate_sub_Hessian_without_penalty(output,
										input,
										subset_num) ==
	  Succeeded::no)
	return Succeeded::no;
    }

  return Succeeded::yes;
}

template <typename TargetT>
Succeeded 
GeneralisedObjectiveFunction<TargetT>::
add_multiplication_with_approximate_Hessian(TargetT& output,
					    const TargetT& input) const
{
  for (int subset_num=0; subset_num<this->get_num_subsets(); ++subset_num)
    {
      if (this->add_multiplication_with_approximate_sub_Hessian(output,
								input,
								subset_num) ==
	  Succeeded::no)
	return Succeeded::no;
    }

  return Succeeded::yes;
}

template <typename TargetT>
Succeeded 
GeneralisedObjectiveFunction<TargetT>::
actual_add_multiplication_with_approximate_sub_Hessian_without_penalty(TargetT& output,
								       const TargetT& input,
								       const int subset_num) const
{
  error("GeneralisedObjectiveFunction:\n"
	"actual_add_multiplication_with_approximate_sub_Hessian_without_penalty implementation is not overloaded by your objective function.");
  return Succeeded::no;
}

/////////////////////// other functions

template <typename TargetT>
std::string
GeneralisedObjectiveFunction<TargetT>::
get_objective_function_values_report(const TargetT& current_estimate)
{
#ifdef BOOST_NO_STRINGSTREAM
  char str[10000];
  ostrstream s(str, 10000);
#else
  std::ostringstream s;
#endif
  const double no_penalty = 
    this->compute_objective_function_without_penalty(current_estimate);
  const double penalty =
    this->compute_penalty(current_estimate);
  s << "Objective function without penalty " << no_penalty
    << "\nPenalty                            " << penalty
    << "\nDifference (i.e. total)            " << no_penalty-penalty
    << '\n';
  return s.str();
}

template<typename TargetT>
bool
GeneralisedObjectiveFunction<TargetT>::
subsets_are_approximately_balanced() const
{
  std::string dummy;
  return this->actual_subsets_are_approximately_balanced(dummy);
}

template<typename TargetT>
bool
GeneralisedObjectiveFunction<TargetT>::
subsets_are_approximately_balanced(std::string& warning_message) const
{
  return this->actual_subsets_are_approximately_balanced(warning_message);
}


#  ifdef _MSC_VER
// prevent warning message on instantiation of abstract class 
#  pragma warning(disable:4661)
#  endif

template class GeneralisedObjectiveFunction<DiscretisedDensity<3,float> >;
template class GeneralisedObjectiveFunction<ParametricVoxelsOnCartesianGrid >; 

END_NAMESPACE_STIR


