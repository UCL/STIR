/*
    Copyright (C) 2003 - 2011-01-14, Hammersmith Imanet Ltd
    Copyright (C) 2012, Kris Thielemans

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup GeneralisedObjectiveFunction
  \brief Declaration of class stir::PoissonLogLikelihoodWithLinearModelForMean

  \author Kris Thielemans
  \author Sanida Mustafovic
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

  This class computes the gradient directly, via \c compute_sub_gradient_without_penalty().
  This method is utilised by the \c OSSPS algorithm in STIR.
  However, an additional method (\c compute_sub_gradient_without_penalty_plus_sensitivity())
  is provided that computes the sum of the subset gradient (without penalty) and the sensitivity.
  This method is utilised by the \c OSMAPOSL algorithm.

  See also \c PoissonLogLikelihoodWithLinearModelForMeanAndListModeData.

  \par Relation with Kullback-Leibler distance

  Note that up to terms independent of \f$\bar y\f$, the log likelihood
  is equal to minus the
  Kullback-Leibler generalised distance

  \f[ \mathrm{KL}(y,\bar y) = \sum_b y_b \mathrm{log}(y_b/\bar y_b) + \bar y_b - y_b \f]

  which has the nice property that \f$\mathrm{KL}(y,\bar y)=0\f$ implies that
  \f$y=\bar y\f$.

  \par Parameters for parsing
  Defaults are indicated below
  \verbatim
  ; specifies if we keep separate sensitivity images (which is more accurate and is
  ; recommended) or if we assume the subsets are exactly balanced
  ; and hence compute the subset-senstivity as sensitivity/num_subsets (this uses less memory).
  use_subset_sensitivities := 1
  ; for recomputing sensitivity, even if a filename is specified
  recompute sensitivity:= 0
  ; filename for reading the sensitivity, or writing if it is recomputed
  ; if use_subset_sensitivities=0
  sensitivity filename:=
  ; pattern for filename for reading the subset sensitivities, or writing if recomputed
  ; if use_subset_sensitivities=1
  ; e.g. subsens_%d.hv
  ; fmt::format is used with the pattern
  subset sensitivity filenames:=
  \endverbatim

  \par Terminology
  We currently use \c sub_gradient for the gradient of the likelihood of the subset (not
  the mathematical subgradient).
*/
template <typename TargetT>
class PoissonLogLikelihoodWithLinearModelForMean : public GeneralisedObjectiveFunction<TargetT>
{
private:
  typedef GeneralisedObjectiveFunction<TargetT> base_type;

public:
  // PoissonLogLikelihoodWithLinearModelForMean();

  //! Compute the subset gradient of the (unregularised) objective function
  /*!
   Implementation in terms of actual_compute_sub_gradient_without_penalty()
   This function is used by OSSPS may be used by other gradient ascent/descent algorithms

    This computes
    \f[
    {\partial L \over \partial \lambda_v} =
      \sum_b P_{bv} ({y_b \over Y_b} - 1)
    \f]
    (see the class general documentation).
    The sum will however be restricted to a subset.
  */
  void compute_sub_gradient_without_penalty(TargetT& gradient, const TargetT& current_estimate, const int subset_num) override;

  //! This should compute the subset gradient of the (unregularised) objective function plus the subset sensitivity
  /*!
   Implementation in terms of actual_compute_sub_gradient_without_penalty().
   This function is used for instance by OSMAPOSL.

    This computes
    \f[ {\partial L \over \partial \lambda_v} + P_v =
      \sum_b P_{bv} {y_b \over Y_b}
      \f]
    (see the class general documentation).
    The sum will however be restricted to a subset.
   */
  virtual void
  compute_sub_gradient_without_penalty_plus_sensitivity(TargetT& gradient, const TargetT& current_estimate, const int subset_num);

  //! set-up sensitivity etc if possible
  /*! If \c recompute_sensitivity is \c false, we will try to
      read it from either \c subset_sensitivity_filenames or \c sensitivity_filename,
      depending on the setting of get_use_subset_sensitivities().

      If \c sensitivity_filename is equal
      to <code>&quot;1&quot;</code>, all data are
      set to \c 1.

      \deprecated This special handling of the string "1" will be removed later.

      Calls set_up_before_sensitivity().
  */
  Succeeded set_up(shared_ptr<TargetT> const& target_sptr) override;

  //! Get a const reference to the total sensitivity
  const TargetT& get_sensitivity() const;
  //! Get a const reference to the sensitivity for a subset
  const TargetT& get_subset_sensitivity(const int subset_num) const;

  //! Add subset sensitivity to existing data
  virtual void add_subset_sensitivity(TargetT& sensitivity, const int subset_num) const = 0;

  //! find out if subset_sensitivities are used
  /*! If \c true, the sub_gradient and subset_sensitivity functions use the sensitivity
      for the given subset, otherwise, we use the total sensitivity divided by the number
      of subsets. The latter uses less memory, but is less stable for most (all?) algorithms.
  */
  bool get_use_subset_sensitivities() const;

  //! find current value of recompute_sensitivity
  bool get_recompute_sensitivity() const;

  //! get filename to read (or write) the total sensitivity
  /*! will be a zero string if not set */
  std::string get_sensitivity_filename() const;
  //! get filename pattern to read (or write) the subset sensitivities
  /*! will be a zero string if not set.
  Could be e.g. "subsens_%d.hv"
  fmt::format is used with the pattern
 */
  std::string get_subsensitivity_filenames() const;

  /*! \name Functions to set parameters
    This can be used as alternative to the parsing mechanism.
   \warning After using any of these, you have to call set_up().
   \warning Be careful with setting shared pointers. If you modify the objects in
   one place, all objects that use the shared pointer will be affected.
  */
  //@{
  void set_recompute_sensitivity(const bool);
  void set_subset_sensitivity_sptr(const shared_ptr<TargetT>&, const int subset_num);

  //! See get_use_subset_sensitivities()
  void set_use_subset_sensitivities(const bool);

  //! set filename to read (or write) the total sensitivity
  /*! set to a zero-length string to avoid reading/writing a file */
  void set_sensitivity_filename(const std::string&);
  //! set filename pattern to read (or write) the subset sensitivities
  /*! set to a zero-length string to avoid reading/writing a file
  Could be e.g. "subsens_%d.hv"
  fmt::format is used with the pattern (which means you can use it like sprintf)

  Calls error() if the pattern is invalid.
 */
  void set_subsensitivity_filenames(const std::string&);
  //@}

  /*! The implementation checks if the sensitivity of a voxel is zero. If so,
   it will the target voxel will be assigned the desired value.
  */
  void fill_nonidentifiable_target_parameters(TargetT& target, const float value) const override;

private:
  std::string sensitivity_filename;
  std::string subsensitivity_filenames;
  bool recompute_sensitivity;
  bool use_subset_sensitivities;

  VectorWithOffset<shared_ptr<TargetT>> subsensitivity_sptrs;
  shared_ptr<TargetT> sensitivity_sptr;

  //! Get the subset sensitivity sptr
  shared_ptr<TargetT> get_subset_sensitivity_sptr(const int subset_num) const;
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
  virtual Succeeded set_up_before_sensitivity(shared_ptr<const TargetT> const& target_sptr) = 0;

  //! compute subset and total sensitivity
  /*! This function fills in the sensitivity data by calling add_subset_sensitivity()
      for all subsets. It assumes that the subsensitivity for the 1st subset has been
      allocated already (and is the correct size).
  */
  void compute_sensitivities();

  //! computes the subset gradient of the objective function without the penalty (optional: add subset sensitivity)
  /*!
    If \c add_sensitivity is \c true, this computes
    \f[ {\partial L \over \partial \lambda_v} + P_v =
      \sum_b P_{bv} {y_b \over Y_b}
      \f]
    (see the class general documentation).
    The sum will however be restricted to a subset.

    However, if \c add_sensitivity is \c false, this function will instead compute only the gradient
    \f[
        {\partial L \over \partial \lambda_v} =
            \sum_b P_{bv} ({y_b \over Y_b} - 1)
    \f]
  */
  virtual void actual_compute_subset_gradient_without_penalty(TargetT& gradient,
                                                              const TargetT& current_estimate,
                                                              const int subset_num,
                                                              const bool add_sensitivity)
      = 0;

  //! Sets defaults for parsing
  /*! Resets \c sensitivity_filename, \c subset_sensitivity_filenames to empty,
     \c recompute_sensitivity to \c false, and \c use_subset_sensitivities to false.
  */
  void set_defaults() override;
  void initialise_keymap() override;
  bool post_processing() override;
};

END_NAMESPACE_STIR

//#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMean.inl"

#endif
