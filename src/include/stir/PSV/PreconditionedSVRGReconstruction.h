//
//
/*
    Copyright (C) 2026 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup PSV
  \brief Declaration of class stir::PreconditionedSVRGReconstruction

  \author Manil Sabeur

  \par Attribution

  This is a native C++ port of the "psv" (preconditioned SVRG) contribution to the
  PETRIC2 reconstruction challenge, by:
  - Hok Shing Wong, University of Bath, UK
  - Margaret Duff, STFC, UK
  - Matthias Ehrhardt, University of Bath, UK
  - Georg Schramm, KU Leuven, Belgium

  The original (PETRIC2-Speedy, branch \c psv, https://github.com/SyneRBI/PETRIC2-Speedy/tree/psv)
  is pure Python running on SIRF+CIL; the algorithm itself is the same as "ALG1" of the MaGeZ
  contribution to PETRIC1. This class reimplements that algorithm on STIR primitives, driven by a
  \c .par file, with no SIRF and no CIL. All credit for the algorithm and its hyper-parameters
  belongs to the original authors; this file only ports it to native STIR.

  \par References

  - M. J. Ehrhardt et al., "Fast PET reconstruction with variance reduction and prior-aware
    preconditioning," Front. Nucl. Med., 2025. https://doi.org/10.3389/fnume.2025.1641215
  - R. Twyman et al., "An Investigation of Stochastic Variance Reduction Algorithms for Relative
    Difference Penalized 3D PET Image Reconstruction," IEEE TMI, 2023.
  - J. Nuyts et al., "A concave prior penalizing relative differences for maximum-a-posteriori
    reconstruction in emission tomography," IEEE TNS, vol. 49, 2002.

*/

#ifndef __stir_PSV_PreconditionedSVRGReconstruction_h__
#define __stir_PSV_PreconditionedSVRGReconstruction_h__

#include "stir/PreconditionedGradient/PreconditionedGradientReconstruction.h"
#include "stir/RegisteredParsingObject.h"
#include <string>

START_NAMESPACE_STIR

/*!
  \ingroup PSV
  \brief Preconditioned Stochastic Variance Reduced Gradient (SVRG) reconstruction: the "psv" preset.

  A \b locked preset of PreconditionedGradientReconstruction reproducing the PETRIC2 "psv" algorithm:
  it fixes the estimator to SVRG and the preconditioner to Hessian-diagonal, and exposes the psv
  hyper-parameters as its own keys. Selecting a different estimator or preconditioner is \em not
  possible here by design; use PreconditionedGradientReconstruction (registered name
  \c "Preconditioned Gradient") for that. The update is

  \f[ \lambda^{k+1} = \left[ \lambda^k + \alpha_k\, P\, \tilde g_k \right]_+ \f]

  with \f$\tilde g\f$ the SVRG variance-reduced gradient and \f$P = m\,(\lambda + \delta\max\lambda_s)/
  (A^\mathrm{T}1 + c\,H_{jj}\,\lambda)\f$ the Hessian-diagonal preconditioner.

  \par Implementation notes

  - The penalisation factor of the prior is used as is: the prior gradient is already divided by
    \c num_subsets inside GeneralisedObjectiveFunction::compute_sub_gradient.
  - \c precond_hessian_factor (\f$c\f$) multiplies only the prior term of the denominator. It
    therefore changes the balance between prior and likelihood curvature voxel by voxel, and
    cannot be absorbed in the step size.
  - \f$H_{jj}\f$ is evaluated on the Gaussian-smoothed image, while the \f$\lambda\f$ multiplying
    it in the denominator is not smoothed.
  - With \c precond_delta_rel equal to 0 (the default, as in the original), \f$P\f$ vanishes where
    \f$\lambda = 0\f$, so a voxel that has been clamped to zero stays at zero. A small positive
    value avoids this.
  - The FOV mask sets \f$P\f$ to zero where \f$A^\mathrm{T}1\f$ is (close to) zero, where
    \f$P\f$ would otherwise be very large.
  - The default step size (3.5) assumes that the prior contributes substantially to the
    denominator of \f$P\f$. With a weak prior, \f$P\f$ approaches the EM preconditioner
    \f$\lambda/A^\mathrm{T}1\f$, whose natural step size is 1, and the default step size can make
    the reconstruction diverge. Reduce <tt>initial step size</tt> in that case.

  \par Attribution

  Native port of the "psv" (preconditioned SVRG) PETRIC2 contribution by Hok Shing Wong
  (Univ. of Bath), Margaret Duff (STFC), Matthias Ehrhardt (Univ. of Bath) and Georg Schramm
  (KU Leuven). All credit for the algorithm and hyper-parameters belongs to the original authors.

  \warning This class should be the last in the Reconstruction hierarchy.
*/
template <class TargetT>
class PreconditionedSVRGReconstruction : public RegisteredParsingObject<PreconditionedSVRGReconstruction<TargetT>,
                                                                        Reconstruction<TargetT>,
                                                                        PreconditionedGradientReconstruction<TargetT>>
{
private:
  typedef RegisteredParsingObject<PreconditionedSVRGReconstruction<TargetT>,
                                  Reconstruction<TargetT>,
                                  PreconditionedGradientReconstruction<TargetT>>
      base_type;

public:
  //! Name used when parsing a Reconstruction object
  static const char* const registered_name;

  //! Default constructor (calls set_defaults())
  PreconditionedSVRGReconstruction();

  //! Construct from a parameter file (or ask_parameters() when filename == "")
  explicit PreconditionedSVRGReconstruction(const std::string& parameter_filename);

  //! builds the psv-configured SVRG estimator + Hessian-diagonal preconditioner, then sets up the engine
  Succeeded set_up(shared_ptr<TargetT> const& target_image_ptr) override;

protected:
  //! @name psv hyper-parameters. They configure the (locked) Hessian-diagonal preconditioner and
  //! SVRG estimator built in set_up(). The step, filter and schedule keys are inherited from the
  //! engine base and re-exposed here; the estimator/preconditioner \em type keys are NOT exposed.
  //! @{
  double precond_hessian_factor;        //!< \f$c\f$ (psv default 2.0)
  double precond_filter_fwhm_mm;        //!< FWHM (mm) of the Gaussian before \f$H_{jj}\f$
  double precond_delta_rel;             //!< keeps \f$P>0\f$ where \f$\lambda=0\f$ (psv default 0)
  int complete_gradient_epoch_interval; //!< SVRG snapshot interval (psv default 2)
  double adjoint_ones_floor;            //!< floor on \f$A^\mathrm{T}1\f$
  std::string fov_mask_filename;        //!< optional FOV mask file; empty -> derived from \f$A^\mathrm{T}1\f$
  double fov_threshold;                 //!< relative threshold to derive the FOV mask
  //! @}

  void set_defaults() override;
  void initialise_keymap() override;
  bool post_processing() override;
};

END_NAMESPACE_STIR

#endif
