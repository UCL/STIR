//
//
/*
    Copyright (C) 2005- 2008, Hammersmith Imanet
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup recon_buildblock
  \brief Implementation of class stir::SumOfGeneralisedObjectiveFunctions

  \author Kris Thielemans

*/
#include "stir/Succeeded.h"
#include "stir/error.h"
#include <algorithm>

START_NAMESPACE_STIR
template <typename ObjFuncT, typename TargetT, typename Parent>
template <typename IterT>
void
SumOfGeneralisedObjectiveFunctions<ObjFuncT, TargetT, Parent>::set_functions(IterT begin, IterT end)
{
  this->_functions.resize(0);
  std::copy(begin, end, this->_functions.begin());
}

template <typename ObjFuncT, typename TargetT, typename Parent>
SumOfGeneralisedObjectiveFunctions<ObjFuncT, TargetT, Parent>::SumOfGeneralisedObjectiveFunctions()
{}

template <typename ObjFuncT, typename TargetT, typename Parent>
template <typename IterT>
SumOfGeneralisedObjectiveFunctions<ObjFuncT, TargetT, Parent>::SumOfGeneralisedObjectiveFunctions(IterT begin, IterT end)
{
  set_functions(begin, end);
}

template <typename ObjFuncT, typename TargetT, typename Parent>
SumOfGeneralisedObjectiveFunctions<ObjFuncT, TargetT, Parent>::~SumOfGeneralisedObjectiveFunctions()
{}

#if 0
// this fails, as _functions might only be balid after set_up
template <typename ObjFuncT, typename TargetT, typename Parent>
TargetT *
SumOfGeneralisedObjectiveFunctions<ObjFuncT, TargetT, Parent>::
construct_target_ptr() const
{
  if (this->_functions.size() == 0)
    return 0;
  else
    return this->_functions[0].construct_target_ptr();
}
#endif

template <typename ObjFuncT, typename TargetT, typename Parent>
Succeeded
SumOfGeneralisedObjectiveFunctions<ObjFuncT, TargetT, Parent>::set_up(shared_ptr<TargetT> const& target_sptr)
{
  if (base_type::set_up(target_sptr) != Succeeded::yes)
    return Succeeded::no;

  _functions_iterator_type iter = this->_functions.begin();
  _functions_iterator_type end_iter = this->_functions.end();
  while (iter != end_iter)
    {
      if (iter->set_up(target_sptr) == Succeeded::no)
        return Succeeded::no;
      ++iter;
    }
  return Succeeded::yes;
}

template <typename ObjFuncT, typename TargetT, typename Parent>
void
SumOfGeneralisedObjectiveFunctions<ObjFuncT, TargetT, Parent>::compute_sub_gradient_without_penalty(
    TargetT& gradient, const TargetT& current_estimate, const int subset_num)
{
  _functions_iterator_type iter = this->_functions.begin();
  _functions_iterator_type end_iter = this->_functions.end();
  while (iter != end_iter)
    {
      iter->compute_sub_gradient_without_penalty(gradient, current_estimate, subset_num);

      ++iter;
    }
}

template <typename ObjFuncT, typename TargetT, typename Parent>
double
SumOfGeneralisedObjectiveFunctions<ObjFuncT, TargetT, Parent>::actual_compute_objective_function_without_penalty(
    const TargetT& current_estimate, const int subset_num)
{
  _functions_iterator_type iter = this->_functions.begin();
  _functions_iterator_type end_iter = this->_functions.end();
  double result = 0;
  while (iter != end_iter)
    {
      result += iter->compute_objective_function_without_penalty(current_estimate, subset_num);

      ++iter;
    }
  return result;
}

template <typename ObjFuncT, typename TargetT, typename Parent>
int
SumOfGeneralisedObjectiveFunctions<ObjFuncT, TargetT, Parent>::set_num_subsets(const int new_num_subsets)
{
  this->num_subsets = new_num_subsets;
  _functions_iterator_type iter = this->_functions.begin();
  _functions_iterator_type end_iter = this->_functions.end();
  while (iter != end_iter)
    {
      // we could check if they all return the same num_subsets, but cannot be bothered now
      if (iter->set_num_subsets(this->num_subsets) != this->num_subsets)
        error("set_num_subsets failed to set to %d subsets", this->num_subsets);
      ++iter;
    }
  return this->num_subsets;
}

template <typename ObjFuncT, typename TargetT, typename Parent>
bool
SumOfGeneralisedObjectiveFunctions<ObjFuncT, TargetT, Parent>::actual_subsets_are_approximately_balanced(
    std::string& warning_message) const
{
  _functions_const_iterator_type iter = this->_functions.begin();
  _functions_const_iterator_type end_iter = this->_functions.end();
  while (iter != end_iter)
    {
      if (!iter->subsets_are_approximately_balanced(warning_message))
        return false;
      ++iter;
    }
  return true;
}

END_NAMESPACE_STIR
