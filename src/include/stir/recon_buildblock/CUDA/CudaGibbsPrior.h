//
//
/*
    Copyright (C) 2025, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details.
*/
/*!
  \file
  \ingroup priors
  \ingroup CUDA
  \brief Declaration of class stir::CudaGibbsPrior

  \author Kris Thielemans
  \author Matteo Colombo
*/

#ifndef __stir_recon_buildblock_CUDA_CudaGibbsPrior_H__
#define __stir_recon_buildblock_CUDA_CudaGibbsPrior_H__

#include "stir/recon_buildblock/GibbsPrior.h"
#include "stir/Array.h"
#include "stir/DiscretisedDensity.h"
#include "stir/shared_ptr.h"
#include "stir/cuda_utilities.h"
#include <string>

// #ifndef __CUDACC__
// #error This file needs to be included when compiling with CUDA
// #endif

START_NAMESPACE_STIR

  /*!
  \ingroup priors
  \ingroup CUDA
  \brief
  A CUDA-accelerated implementation of the Gibbs prior inheriting from GibbsPrior.

  This class provides CUDA-accelerated implementations of compute_value, compute_gradient, 
  and accumulate_Hessian_times_input while inheriting all parameter parsing and 
  CPU functionality from the base GibbsPrior class.
  
  The potential function is provided via the template parameter PotentialFun, which should
  provide device functions for value(), derivative_10(), derivative_11(), and derivative_20().

  The prior is computed as follows:
  \f[
  f = \sum_{r,dr} w_{dr} \psi(\lambda_r, \lambda_{r+dr}) * \kappa_r * \kappa_{r+dr}
  \f]
  with gradient
  \f[
  g_r = \sum_{dr} w_{dr} \psi'(\lambda_r, \lambda_{r+dr}) * \kappa_r * \kappa_{r+dr}
  \f]
  where \f$\lambda\f$ is the image, \f$\psi\f$ is the potential function, \f$r\f$ and \f$dr\f$ are indices 
  and the sum is over the neighbourhood where the weights \f$w_{dr}\f$ are non-zero.

  The \f$\kappa\f$ image can be used to have spatially-varying penalties such as in
  Jeff Fessler's papers. It should have identical dimensions to the image for which the
  penalty is computed. If \f$\kappa\f$ is not set, this class will effectively
  use 1 for all \f$\kappa\f$'s.

  All parsing functionality is inherited from GibbsPrior - see GibbsPrior documentation
  for parameter keywords.
*/

template <typename elemT, typename CudaPotentialT>
class CudaGibbsPrior : public GibbsPrior<elemT, CudaPotentialT>
{
private:
  typedef GibbsPrior<elemT, CudaPotentialT> base_type;

protected:
  //! CUDA grid and block dimensions
  dim3 block_dim;
  dim3 grid_dim;

  elemT *d_image_data, *d_gradient_data;

  int3 d_Image_dim;
  int3 d_Image_max_indices;
  int3 d_Image_min_indices;
  int3 d_weight_max_indices;
  int3 d_weight_min_indices;

  float* d_weights_data = nullptr;
  elemT* d_kappa_data = nullptr;
  CudaPotentialT potential;
  
public:

  CudaGibbsPrior();
  CudaGibbsPrior(const bool only_2D, float penalization_factor);
  ~CudaGibbsPrior();

  //! Override to set up CUDA resources and call parent set_up
  Succeeded set_up(shared_ptr<const DiscretisedDensity<3, elemT>> const& target_sptr) override;
  
  void set_weights(const Array<3, float>& w) override;
  
  void set_kappa_sptr(const shared_ptr<const DiscretisedDensity<3, elemT>>& k) override;

  //! CUDA-accelerated value computation
  double compute_value(const DiscretisedDensity<3, elemT>& current_image_estimate) override;

  //! CUDA-accelerated gradient computation
  void compute_gradient(DiscretisedDensity<3, elemT>& prior_gradient,
                        const DiscretisedDensity<3, elemT>& current_image_estimate) override;
  
  double compute_gradient_times_input(const DiscretisedDensity<3, elemT>& input,
                                    const DiscretisedDensity<3, elemT>& current_image_estimate) override;

  void compute_Hessian_diagonal(DiscretisedDensity<3, elemT>& Hessian_diagonal, const DiscretisedDensity<3, elemT>& current_estimate) const override;


  //! CUDA-accelerated Hessian times input computation
  void accumulate_Hessian_times_input(DiscretisedDensity<3, elemT>& output,
                                      const DiscretisedDensity<3, elemT>& current_estimate,
                                      const DiscretisedDensity<3, elemT>& input) const override;
                      
};

END_NAMESPACE_STIR

#ifdef __CUDACC__
#include "stir/recon_buildblock/CUDA/CudaGibbsPrior.cuh"
#endif



#endif // __stir_recon_buildblock_CUDA_CudaGibbsPrior_H__
