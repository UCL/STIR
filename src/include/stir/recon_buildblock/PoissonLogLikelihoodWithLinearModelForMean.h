//
// $Id$
//
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd

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
  \brief Declaration of class stir::PoissonLogLikelihoodWithLinearModelForMean

  \author Kris Thielemans
  \author Sanida Mustafovic

  $Date$
  $Revision$
*/

#ifndef __stir_recon_buildblock_PoissonLogLikelihoodWithLinearModelForMean_H__
#define __stir_recon_buildblock_PoissonLogLikelihoodWithLinearModelForMean_H__

#include "stir/recon_buildblock/GeneralisedObjectiveFunction.h"

START_NAMESPACE_STIR


/*!
  \ingroup GeneralisedObjectiveFunction
  \brief a base class for  LogLikelihood of independent Poisson variables 
  where the mean values are linear combinations of the parameters.

  Suppose the measured data \f$y\f$ are statistically independent and
  Poisson distributed with mean \f$\bar y\f$. The log likelihood is then 
  given by

  \f[ L(y\;\bar y) = \sum_b y_b log \bar y_b - \mathrm{log} y_b! - \bar y_b \f]
 
  If the means are modeled by some functions of some parameters
  \f$Y(\lambda)\f$, the gradient w.r.t. those parameters follows 
  immediately as

  \f[ {\partial L \over \partial \lambda_v} =
      \sum_b \left( {y_b \over Y_b} - 1 \right) 
             {\partial Y_b \over \partial \lambda_v }
  \f]

  In this class, it is assumed that the means are linear functions of
  the parameters, which using matrices and vectors can be written
  (in a notation familiar to PET people) as

  \f[ Y = P \lambda + r \f]

  If the background term \f$r\f$ is 0, \f$P_{bv}\f$  gives
  the conditional probability of detecting a count in bin \f$b\f$
  if the parameter \f$\lambda_v\f$ is 1 and all others 0.

  In this case, the gradient is given by
  \f[ {\partial L \over \partial \lambda_v} =
      \sum_b P_{bv} {y_b \over Y_b} - P_v
  \f]
  where 
  \f[ 
  P_v = \sum_b P_{bv}
  \f]
  where the sum is over all possible bins (not only those where
  any counts were detected). We call the vector given by \f$P_v\f$
  the <i>sensitivity</i> because (if \f$r=0\f$) it is the total
  probability of detecting a count (in any bin) originating from \f$v\f$.

  This class computes the gradient as a sum of these two terms. The
  sensitivity has to be computed by the virtual function 
  \c add_subset_sensitivity(). The sum is computed by
  \c compute_sub_gradient_without_penalty_plus_sensitivity().

  The reason for this is that the sensitivity is data-independent, and
  can be computed only once. See also
  PoissonLogLikelihoodWithLinearModelForMeanAndListModeData.
  
  \par Relation with Kullback-Leibler distance

  Note that up to terms independent of \f$\bar y\f$, the log likelihood
  is equal to minus the
  Kullback-Leibler generalised distance

  \f[ \mathrm{KL}(y,\bar y) = \sum_b y_b \mathrm{log}(y_b/\bar y_b) + \bar y_b - y_b \f]

  which has the nice property that \f$\mathrm{KL}(y,\bar y)=0\f$ implies that 
  \f$y=\bar y\f$.

  \par Parameters for parsing
  Defaults are indicate below
  \verbatim
  ; specifies if we keep separate sensitivity images (which is more accurate and is 
  ; recommended) or if we assume the subsets are exactly balanced (this uses more memory).
  use_subset_sensitivities := 0
  ; for recomputing sensitivity, even if a filename is specified
  recompute sensitivity:= 0
  ; filename for reading the sensitivity, or writing if it is recomputed
  ; if use_subset_sensitivities=0
  sensitivity filename:=
  ; pattern for filename for reading the subset sensitivities, or writing if recomputed
  ; if use_subset_sensitivities=1
  ; e.g. subsens_%d.hv
  ; boost::format is used with the pattern (which means you can use it like sprintf)
  subset sensitivity filenames:=
  \endverbatim


*/
template <typename TargetT>
class PoissonLogLikelihoodWithLinearModelForMean: 
public  GeneralisedObjectiveFunction<TargetT>
{
 private:
  typedef GeneralisedObjectiveFunction<TargetT> base_type;

 public:
  
  //PoissonLogLikelihoodWithLinearModelForMean(); 

  //! Implementation in terms of compute_sub_gradient_without_penalty_plus_sensitivity()
  /*! \warning If separate subsensitivities are not used, we just subtract the total 
    sensitivity divided by the number of subsets.
    This is fine for some algorithms as the sum over all the subsets is 
    equal to gradient of the objective function (without prior). 
    Other algorithms do not behave very stable under this approximation
    however. So, currently setup() will return an error if
    <code>!subsets_are_approximately_balanced()</code> and subset sensitivities
    are not used.

    \see get_use_subset_sensitivities()
  */
  virtual void 
    compute_sub_gradient_without_penalty(TargetT& gradient, 
                                         const TargetT &current_estimate, 
                                         const int subset_num); 

  //! This should compute the gradient of the (unregularised) objective function plus the (sub)sensitivity
  /*! 
    This function is used for instance by OSMAPOSL.

    This computes
    \f[ {\partial L \over \partial \lambda_v} + P_v =
      \sum_b P_{bv} {y_b \over Y_b}
      \f]
    (see the class general documentation).
    The sum will however be restricted to a subset.
   */
  virtual void 
    compute_sub_gradient_without_penalty_plus_sensitivity(TargetT& gradient, 
                                                          const TargetT &current_estimate, 
                                                          const int subset_num) =0; 

  //! set-up sensitivity etc if possible
  /*! If \c recompute_sensitivity is \c false, we will try to
      read it from either \c subset_sensitivity_filenames or \c sensitivity_filename, 
      depending on the setting of get_use_subset_sensitivities().

      If \c sensitivity_filename is equal
      to <code>&quot;1&quot;</code>, all data are
      set to \c 1.

      \warning The special handling of the string \c might be removed later.

      Calls set_up_before_sensitivity().
  */
  virtual Succeeded set_up(shared_ptr <TargetT> const& target_sptr);

  //! Get a const reference to the total sensitivity
  const TargetT& get_sensitivity() const;
  //! Get a const reference to the sensitivity for a subset
  const TargetT& get_subset_sensitivity(const int subset_num) const;

  //! Add subset sensitivity to existing data
  virtual void
    add_subset_sensitivity(TargetT& sensitivity, const int subset_num) const = 0;

  //! find out if subset_sensitivities are used
  /*! If \c true, the sub_gradient and subset_sensitivity functions use the sensitivity
      for the given subset, otherwise, we use the total sensitivity divided by the number 
      of subsets. The latter uses less memory, but is less stable for most (all?) algorithms.
  */
  bool get_use_subset_sensitivities() const;

  //! find current value of recompute_sensitivity
  bool get_recompute_sensitivity() const;

  /*! \name Functions to set parameters
    This can be used as alternative to the parsing mechanism.
   \warning After using any of these, you have to call set_up().
   \warning Be careful with setting shared pointers. If you modify the objects in 
   one place, all objects that use the shared pointer will be affected.
  */
  //@{
  void set_recompute_sensitivity(const bool);
  void set_subset_sensitivity_sptr(const shared_ptr<TargetT>&, const int subset_num);

  void set_use_subset_sensitivities(const bool);
  //@}

  /*! The implementation checks if the sensitivity of a voxel is zero. If so,
   it will the target voxel will be assigned the desired value.
  */
  void 
    fill_nonidentifiable_target_parameters(TargetT& target, const float value ) const;

 private:

  std::string sensitivity_filename;
  std::string subsensitivity_filenames;
  bool recompute_sensitivity;
  bool use_subset_sensitivities;

  VectorWithOffset<shared_ptr<TargetT> > subsensitivity_sptrs;
  shared_ptr<TargetT> sensitivity_sptr;

  //! Get the subset sensitivity sptr
  shared_ptr<TargetT> 
    get_subset_sensitivity_sptr(const int subset_num) const;
  //! compute total from subsensitivity or vice versa
  /*! This function will be called by set_up() after reading new images, and/or 
      by compute_sensitivities().

      if get_use_subset_sensitivities() is true, the total sensitivity is computed
      by adding the subset sensitivities, otherwise, the subset sensitivities are
      computed by dividng the total sensitivity by \c num_subsets.
  */
  void set_total_or_subset_sensitivities();

protected:
  //! set-up specifics for the derived class 
  virtual Succeeded 
    set_up_before_sensitivity(shared_ptr<TargetT > const& target_sptr) = 0;

  //! compute subset and total sensitivity
  /*! This function fills in the sensitivity data by calling add_subset_sensitivity()
      for all subsets. It assumes that the subsensitivity for the 1st subset has been 
      allocated already (and is the correct size).
  */
  void compute_sensitivities();

  //! Sets defaults for parsing 
  /*! Resets \c sensitivity_filename, \c subset_sensitivity_filenames to empty,
     \c recompute_sensitivity to \c false, and \c use_subset_sensitivities to false.
  */
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

};

END_NAMESPACE_STIR

//#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMean.inl"

#endif
