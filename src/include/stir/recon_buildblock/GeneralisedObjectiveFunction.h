//
//
/*
    Copyright (C) 2003- 2009, Hammersmith Imanet Ltd
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

*/
#ifndef __stir_recon_buildblock_GeneralisedObjectiveFunction_H__
#define __stir_recon_buildblock_GeneralisedObjectiveFunction_H__


#include "stir/RegisteredObject.h"
#include "stir/ParsingObject.h"
#include "stir/shared_ptr.h"
#include "stir/recon_buildblock/GeneralisedPrior.h"
#include <string>

#include <stir/ExamInfo.h>

START_NAMESPACE_STIR


class Succeeded;

/*!
  \ingroup GeneralisedObjectiveFunction
  \brief
  A base class for 'generalised' objective functions, i.e. objective
  functions for which at least a 'gradient' is defined. 

  Some iterative algorithms use an 'objective function' only in a 
  loose sense. They might for instance allow generalisations 
  which no longer optimise a function. For example in the case
  of PoissonLogLikelihoodWithLinearModelForMeanAndProjData
  with non-matching forward and back projectors, the 'gradient' 
  that is computed is generally not the gradient of the
  log-likelihood that corresponds to the forward projector.
  However, one hopes that it still points towards the optimum.

  Often, one includes a penalty (or prior) in the objective function. This
  class uses a GeneralisedPrior object for this. Note that we use the convention that
  the objective function is maximised. The penalty is expected to
  be a function that increases with higher penalty, so it will be <i>subtracted</i>
  from the unregularised case.

  In tomography, we often use subsets, where the objective function
  is written as a sum of sub-objective functions. This class has some
  subset functionality. When using subsets, the 
  penalty will be distributed evenly over all subsets. While this increases
  the computational cost, it makes the subsets more 'balanced' which
  is best for most algorithms.

  \see IterativeReconstruction


  \todo Currently, there is subset code in both IterativeReconstruction and here.
  This is confusing and leads to repetition. It probably should all be moved here.

  \par Parameters for parsing

  \verbatim
  ; specify prior, see GeneralisedObjectiveFunction<TargetT> hierarchy for possible values
  prior type :=
  \endverbatim
*/
template <typename TargetT>
class GeneralisedObjectiveFunction: 
   public RegisteredObject<GeneralisedObjectiveFunction<TargetT> >,
   public ParsingObject
{
public:
  
  //GeneralisedObjectiveFunction(); 

  virtual ~GeneralisedObjectiveFunction(); 


  //! Creates a suitable target as determined by the parameters
  virtual TargetT *
    construct_target_ptr() const = 0; 

  //! Has to be called before using this object
  virtual Succeeded 
    set_up(shared_ptr<TargetT> const& target_sptr);

  //! This should compute the sub-gradient of the objective function at the \a current_estimate
  /*! The subgradient is the gradient of the objective function restricted to the
      subset specified. What this means depends on how this function is implemented later
      on in the hierarchy.

      Computed as the <i>difference</i> of 
      <code>compute_sub_gradient_without_penalty</code>
      and 
      <code>get_prior_ptr()-&gt;compute_gradient()/num_subsets</code>.

    \warning Any data in \a gradient will be overwritten.
  */
  virtual void 
    compute_sub_gradient(TargetT& gradient, 
			 const TargetT &current_estimate, 
			 const int subset_num); 

  //! This should compute the sub-gradient of the unregularised objective function at the \a current_estimate
  /*!     
    \warning The derived class should overwrite any data in \a gradient.
  */
  virtual void 
    compute_sub_gradient_without_penalty(TargetT& gradient, 
					 const TargetT &current_estimate, 
					 const int subset_num) =0; 

  //! Compute the value of the unregularised sub-objective function at the \a current_estimate
  /*! Implemented in terms of actual_compute_objective_function_without_penalty. */
  virtual double
    compute_objective_function_without_penalty(const TargetT& current_estimate,
					       const int subset_num);

  //! Compute the value of the unregularised objective function at the \a current_estimate
  /*! Computed by summing over all subsets.
   */
  virtual double
    compute_objective_function_without_penalty(const TargetT& current_estimate);

  //! Compute the value of the sub-penalty at the \a current_estimate
  /*! As each subset contains the same penalty, this function returns
      the same as 
      \code
      compute_penalty(current_estimate)/num_subsets 
      \endcode
      Implemented in terms of GeneralisedPrior::compute_value.
      \see compute_objective_function(const TargetT&) for sign conventions.
  */
  double
    compute_penalty(const TargetT& current_estimate,
		    const int subset_num);
  //! Compute the value of the penalty at the \a current_estimate
  /*! Implemented in terms of GeneralisedPrior::compute_value. */
  double
    compute_penalty(const TargetT& current_estimate);

  //! Compute the value of the sub-objective function at the \a current_estimate
  /*! Computed as the <i>difference</i> of 
      <code>compute_objective_function_without_penalty</code>
      and 
      <code>compute_penalty</code>.
  */  
  double
    compute_objective_function(const TargetT& current_estimate,
			       const int subset_num);

  //! Compute the value of the objective function at the \a current_estimate
  /*! Computed as the <i>difference</i> of 
      <code>compute_objective_function_without_penalty</code>
      and 
      <code>compute_penalty</code>.
  */  
  double
    compute_objective_function(const TargetT& current_estimate);

  //! Fill any elements that we cannot estimate with a fixed value
  /*! In many cases, it is easier to use a larger target than what we can
    actually estimate. For instance, using a rectangular image while we estimate
    only a circular region.

    For some algorithms, it is important that the parameters that cannot be
    estimate are set to 0 (or some other value). For example, if the outer voxels
    contribute to the forward projection of an image, but not to a backprojection.

    This function allows you to do that. Its default implementation is to do nothing.
    It is up to the derived class to implement this sensible.

    \todo The type of the value should really be derived from e.g. TargetT::full_iterator.
  */
  virtual void 
    fill_nonidentifiable_target_parameters(TargetT& target, const float value ) const
  {}

  //! \name multiplication with (sub)Hessian
  /*! \brief Functions that multiply the (sub)Hessian with a \'vector\'.
      
      All these functions add their result to any existing data in \a output.

      They all call actual_add_multiplication_with_approximate_sub_Hessian_without_penalty.
  */
  //@{
  Succeeded 
      add_multiplication_with_approximate_sub_Hessian_without_penalty(TargetT& output,
								      const TargetT& input,
								      const int subset_num) const;
  Succeeded 
    add_multiplication_with_approximate_sub_Hessian(TargetT& output,
						    const TargetT& input,
						    const int subset_num) const;
  Succeeded 
    add_multiplication_with_approximate_Hessian_without_penalty(TargetT& output,
								const TargetT& input) const;
  Succeeded 
    add_multiplication_with_approximate_Hessian(TargetT& output,
						const TargetT& input) const;
  //@}

  //! Construct a string with info on the value of objective function with and without penalty
  std::string
    get_objective_function_values_report(const TargetT& current_estimate);

  //! Return the number of subsets in-use
  int get_num_subsets() const;


  //! Attempts to change the number of subsets. 
  /*! \return The number of subsets that will be used later, which is not
      guaranteed to be what you asked for. */
  virtual int set_num_subsets(const int num_subsets) = 0;

  //! Checks of the current subset scheme is approximately balanced
  /*! Balanced subsets means that the sub-gradients point all roughly in the
      same direction (at least when far from the optimum).

      This function tests if this is approximately true, such that a reconstruction
      algorithm can either adapt or abort.
     
     Implemented in terms of actual_subsets_are_approximately_balanced(std::string&).
  */
  bool subsets_are_approximately_balanced() const;
  //! Checks of the current subset scheme is approximately balanced and constructs a warning message
  /*! 
    \see subsets_are_approximately_balanced()
    \param  warning_message A string variable. If the subsets are not (approx.)
       balanced, this function will <strong>append</strong>
       a warning message explaining why.
  */
  bool subsets_are_approximately_balanced(std::string& warning_message) const;

  //! check if the prior is set (or the penalisation factor is 0)
  bool prior_is_zero() const;

  //! Read-only access to the prior
  /*! \todo It would be nicer to not return a pointer.
  */
  GeneralisedPrior<TargetT> * const
    get_prior_ptr() const;

  shared_ptr<GeneralisedPrior<TargetT> >
	  get_prior_sptr();

  //! Change the prior
  /*! \warning You should call set_up() again after using this function.
   */
  void set_prior_sptr(const shared_ptr<GeneralisedPrior<TargetT> >&);

  //!
  //! \brief set_input_data
  //! \author Nikos Efthimiou
  //! \details It can be used to set the data to be reconstucted in
  //! real-time ( withint some other code ).
  virtual void set_input_data(const shared_ptr< ExamInfo > &) = 0;

protected:
  int num_subsets;

  shared_ptr<GeneralisedPrior<TargetT> > prior_sptr;

  //! sets any default values
  /*! Has to be called by set_defaults in the leaf-class */
  virtual void set_defaults();
  //! sets parsing keys
  /*! Has to be called by initialise_keymap in the leaf-class */
  virtual void initialise_keymap();

  //virtual bool post_processing();

  //! Implementation of function that checks subset balancing
  /*!
     \see subsets_are_approximately_balanced(std::string&)

     \par Developer\'s note

     The reason we have this function is that overloading 
     subsets_are_approximately_balanced(std::string&) in a derived class
     would hide subsets_are_approximately_balanced().
  */
  virtual bool actual_subsets_are_approximately_balanced(std::string& warning_message) const = 0;

  //! Implementation of function that computes the objective function for the current subset
  /*!
     \see compute_objective_function_without_penalty(const Target&,const int)

     \par Developer\'s note

     The reason we have this function is that overloading a function 
     in a derived class, hides all functions of the 
     same name.
  */
  virtual double
    actual_compute_objective_function_without_penalty(const TargetT& current_estimate,
						      const int subset_num) = 0;

  //! Implementation of the function that multiplies the sub-Hessian with a vector.
  /*!
     \see multiplication_with_approximate_sub_Hessian_without_penalty(TargetT&,const TargetT&, const int).

     \warning The default implementation just calls error(). This behaviour has to be
     overloaded by the derived classes.

     \par Developer\'s note

     The reason we have this function is that overloading a function 
     in a derived class, hides all functions of the 
     same name.
  */
  virtual Succeeded 
      actual_add_multiplication_with_approximate_sub_Hessian_without_penalty(TargetT& output,
								      const TargetT& input,
								      const int subset_num) const;
};

END_NAMESPACE_STIR

#endif
