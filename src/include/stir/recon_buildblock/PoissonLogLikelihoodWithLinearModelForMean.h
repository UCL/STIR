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

  \verbatim
  ; for recomputing sensitivity, even if a filename is specified
  recompute sensitivity:= 0
  ; filename for reading the sensitivity, or writing if it is recomputed
  sensitivity filename:=
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
  /*! \warning At present, we do not keep separate subsensitivities for
    each subset, but just subtract the total sensitivity divided by the
    number of subsets.
    This is fine for some algorithms as the sum over all the subsets is 
    equal to gradient of the objective function (without prior). 
    Other algorithms do not behave very stable under this approximation
    however. So, currently we call error() if
    <code>!subsets_are_approximately_balanced()</code>.
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

  //! set-up sensitivity if possible
  /*! If \c recompute_sensitivity is \c false, we will try to
      read it from \c sensitivity_filename, unless this is equal
      to <code>&quot;1&quot;</code> in which case all data are
      set to \c 1.

      \warning The special handling of the string \c might be removed later.
      \warning This function does not set \c sensitivity_sptr even if
      recomputation is forced. The reason for this is that the derived
      class probably needs to set-up some variables before
      add_subset_sensitivity() can work. 
  */
  virtual Succeeded set_up(shared_ptr <TargetT> const& target_sptr);

  //! Get a const reference to the sensitivity
  const TargetT& get_sensitivity(const int subset_num) const;

  //! Add subset sensitivity to existing data
  virtual void
    add_subset_sensitivity(TargetT& sensitivity, const int subset_num) const = 0;

  bool get_use_subset_sensitivities() const;

  /*! \name Functions to set parameters
    This can be used as alternative to the parsing mechanism.
   \warning After using any of these, you have to call set_up().
   \warning Be careful with setting shared pointers. If you modify the objects in 
   one place, all objects that use the shared pointer will be affected.
  */
  //@{
  void set_recompute_sensitivity(const bool);
  void set_sensitivity_sptr(const shared_ptr<TargetT>&, const int subset_num);

  void set_use_subset_sensitivities(const bool);
  //@}

  /*! The implementation checks if the sensitivity of a voxel is zero. If so,
   it will the target voxel will be assigned the desired value.

   \todo The current implementation uses only get_sensitivity(0);
  */
  void 
    fill_nonidentifiable_target_parameters(TargetT& target, const float value ) const;

protected:

  std::string sensitivity_filename;
  bool recompute_sensitivity;
  bool use_subset_sensitivities;

  VectorWithOffset<shared_ptr<TargetT> > sensitivity_sptrs;

  //! Get the subset sensitivity sptr
  shared_ptr<TargetT> 
    get_subset_sensitivity_sptr(const int subset_num) const;

  void compute_sensitivities();

  //! Sets defaults for parsing 
  /*! Resets \c sensitivity_filename and \c sensitivity_sptr and
     \c recompute_sensitivity to \c false.
  */
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

};

END_NAMESPACE_STIR

//#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMean.inl"

#endif
