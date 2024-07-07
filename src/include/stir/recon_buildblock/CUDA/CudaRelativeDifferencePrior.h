/*
    Copyright (C) 2024, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_recon_buildblock_CudaRelativeDifferencePrior_h__
#define __stir_recon_buildblock_CudaRelativeDifferencePrior_h__

/*!
  \file
  \ingroup priors
  \ingroup CUDA
  \brief implementation of the stir::CudaRelativeDifferencePrior class

  \author Imraj Singh
  \author Kris Thielemans
*/

#include "stir/recon_buildblock/RelativeDifferencePrior.h"

START_NAMESPACE_STIR

struct cppdim3
{
  int x;
  int y;
  int z;
};

/*!
  \ingroup priors
  \ingroup CUDA
  \brief CUDA implementation of the Relative Difference prior

  \see RelativeDifferencePrior. Results should be identical.

  \todo Limitation: currently only weights of size 3x3x3 are supported. Therefore, single slice
  images will lead to an error being thrown.
*/
template <typename elemT>
class CudaRelativeDifferencePrior : public RegisteredParsingObject<CudaRelativeDifferencePrior<elemT>,
                                                                   GeneralisedPrior<DiscretisedDensity<3, elemT>>,
                                                                   RelativeDifferencePrior<elemT>>
{
#ifdef SWIG
public:
#endif
  typedef RegisteredParsingObject<CudaRelativeDifferencePrior<elemT>,
                                  GeneralisedPrior<DiscretisedDensity<3, elemT>>,
                                  RelativeDifferencePrior<elemT>>
      base_type;

public:
  // Name which will be used when parsing a GeneralisedPrior object
  inline static const char* const registered_name = "Cuda Relative Difference Prior";
#ifndef SWIG
  // import constructors from base_type
  // Note: currently disabled for SWIG as that needs SWIG 4.2
  using base_type::base_type;
#endif

  ~CudaRelativeDifferencePrior();

  // Overridden methods
  void set_defaults() override;
  double compute_value(const DiscretisedDensity<3, elemT>& current_image_estimate) override;
  void compute_gradient(DiscretisedDensity<3, elemT>& prior_gradient,
                        const DiscretisedDensity<3, elemT>& current_image_estimate) override;

  Succeeded set_up(shared_ptr<const DiscretisedDensity<3, elemT>> const& target_sptr) override;

protected:
  int z_dim, y_dim, x_dim;
  cppdim3 block_dim;
  cppdim3 grid_dim;

  //! Device copy of weights
  float* d_weights_data = 0;
  //! Device copy of kappa
  float* d_kappa_data = 0;
};

END_NAMESPACE_STIR

#endif // __stir_recon_buildblock_CudaRelativeDifferencePrior_h__
