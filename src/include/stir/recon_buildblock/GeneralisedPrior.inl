//
//
/*!
  \file
  \ingroup priors
  \brief Inline implementations for class stir::GeneralisedPrior

  \author Kris Thielemans
  \author Sanida Mustafovic

*/
/*
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

START_NAMESPACE_STIR

template <typename elemT>
GeneralisedPrior<elemT>::GeneralisedPrior()
{
  penalisation_factor = 0;
}

template <typename elemT>
float
GeneralisedPrior<elemT>::get_penalisation_factor() const
{
  return penalisation_factor;
}

/*!
  \warning Currently we allow the penalisation factor to be set \b after calling set_up().
*/
template <typename elemT>
void
GeneralisedPrior<elemT>::set_penalisation_factor(const float new_penalisation_factor)
{
  penalisation_factor = new_penalisation_factor;
}

template <typename TargetT>
void
GeneralisedPrior<TargetT>::initialise_keymap()
{
  this->parser.add_key("penalisation factor", &this->penalisation_factor);
}

template <typename TargetT>
void
GeneralisedPrior<TargetT>::set_defaults()
{
  _already_set_up = false;
  this->penalisation_factor = 0;
}

template <typename TargetT>
Succeeded
GeneralisedPrior<TargetT>::set_up(shared_ptr<const TargetT> const&)
{
  _already_set_up = true;
  return Succeeded::yes;
}

template <typename TargetT>
double
GeneralisedPrior<TargetT>::compute_gradient_times_input(const TargetT& input, const TargetT& current_estimate)
{
  error("GeneralisedPrior:\n  compute_gradient_times_input is not implemented by your prior.");
  return 0;
}

template <typename TargetT>
void
GeneralisedPrior<TargetT>::compute_Hessian(TargetT& output,
                                           const BasicCoordinate<3, int>& coords,
                                           const TargetT& current_image_estimate) const
{
  if (this->is_convex())
    error("GeneralisedPrior:\n  compute_Hessian implementation is not overloaded by your convex prior.");
  else
    error("GeneralisedPrior:\n  compute_Hessian is not implemented for this (non-convex) prior.");
}

template <typename TargetT>
void
GeneralisedPrior<TargetT>::compute_Hessian_diagonal(TargetT& Hessian_diagonal, const TargetT& current_estimate) const
{
  error("GeneralisedPrior:\n  compute_Hessian_diagonal is not implemented by your prior.");
}

template <typename TargetT>
void
GeneralisedPrior<TargetT>::add_multiplication_with_approximate_Hessian(TargetT& output, const TargetT& input) const
{
  error("GeneralisedPrior:\n"
        "add_multiplication_with_approximate_Hessian implementation is not overloaded by your prior.");
}

template <typename TargetT>
void
GeneralisedPrior<TargetT>::accumulate_Hessian_times_input(TargetT& output,
                                                          const TargetT& current_estimate,
                                                          const TargetT& input) const
{
  error("GeneralisedPrior:\n"
        "accumulate_Hessian_times_input implementation is not overloaded by your prior.");
}

template <typename TargetT>
void
GeneralisedPrior<TargetT>::check(TargetT const& current_estimate) const
{
  if (!_already_set_up)
    error("The prior should already be set-up, but it's not.");
}

END_NAMESPACE_STIR
