//
//
/*
    Copyright (C) 2005- 2008, Hammersmith Imanet
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
  \ingroup recon_buildblock
  \brief Declaration of class stir::SumOfGeneralisedObjectiveFunctions

  \author Kris Thielemans

*/

#ifndef __stir_recon_buildblock_SumOfGeneralisedObjectiveFunctions_H__
#define __stir_recon_buildblock_SumOfGeneralisedObjectiveFunctions_H__


#include "stir/shared_ptr.h"
#include "stir/recon_buildblock/GeneralisedObjectiveFunction.h"
#include <vector>

START_NAMESPACE_STIR

/*!
  \ingroup recon_buildblock
  \brief
  A base class for sums of 'generalised' objective functions, i.e. objective
  functions for which at least a 'gradient' is defined. 
  \todo document why use of ParentT template
  \todo doc subsets
*/
template <typename ObjFuncT,
          typename TargetT, 
          typename ParentT = GeneralisedObjectiveFunction<TargetT> >
class SumOfGeneralisedObjectiveFunctions :
  public ParentT
{
  typedef ParentT base_type;
  typedef SumOfGeneralisedObjectiveFunctions<ObjFuncT, TargetT, ParentT> self_type;
public:
  
  inline 
  SumOfGeneralisedObjectiveFunctions(); 

  template <typename IterT>
  inline
  SumOfGeneralisedObjectiveFunctions(IterT begin, IterT end);
  
  inline virtual 
  ~SumOfGeneralisedObjectiveFunctions(); 

  template <typename IterT>
  inline
  void set_functions(IterT begin, IterT end);

#if 0
  //! Creates a suitable target as determined by the parameters
  /*! \return construct_target_ptr() of the first term in the sum, or 
        0 if there are no terms in the sum.
   */
  inline virtual
  TargetT *
    construct_target_ptr() const; 
#endif

  //! Has to be called before using this object
  /*! Will call set_up() for all terms in the sum, but will stop as soon as
      one set_up() fails.
  */
  inline virtual
  Succeeded 
    set_up(shared_ptr<TargetT> const& target_sptr);

  //! This computes the gradient of the unregularised objective function at the \a current_estimate
  /*! It is computed as the sum of the subgradients for each term, depending on the subset scheme.
  */
  inline virtual
  void 
    compute_sub_gradient_without_penalty(TargetT& gradient, 
					 const TargetT &current_estimate, 
					 const int subset_num); 

  inline virtual
  double 
  actual_compute_objective_function_without_penalty(const TargetT& current_estimate,
						    const int subset_num);

  //! Attempts to change the number of subsets. 
  /*! \return The number of subsets that will be used later, which is not
      guaranteed to be what you asked for. */
  inline virtual
  int set_num_subsets(const int num_subsets) ;

protected:

  typedef std::vector<ObjFuncT> _functions_type;
  typedef typename _functions_type::iterator _functions_iterator_type;
  typedef typename _functions_type::const_iterator _functions_const_iterator_type;
  _functions_type _functions;
  //! Implementation of function that checks subset balancing
  /*!
     \todo doc subset
  */
  inline virtual
  bool actual_subsets_are_approximately_balanced(std::string& warning_message) const;
};

END_NAMESPACE_STIR

#include "stir/recon_buildblock/SumOfGeneralisedObjectiveFunctions.inl"

#endif
