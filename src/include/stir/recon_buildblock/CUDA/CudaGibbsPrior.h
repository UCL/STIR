/*
    Copyright (C) 2025, University College London
    Copyright (C) 2025, University of Milano-Bicocca

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup priors
  \ingroup CUDA
  \brief Declaration of the stir::CudaGibbsPrior class

  \author Kris Thielemans
  \author Matteo Neel Colombo
*/

#ifndef __stir_recon_buildblock_CUDA_CudaGibbsPrior_H__
#define __stir_recon_buildblock_CUDA_CudaGibbsPrior_H__

#include "stir/recon_buildblock/GibbsPrior.h"
#include "stir/Array.h"
#include "stir/DiscretisedDensity.h"
#include <cuda_runtime.h>

#include "stir/shared_ptr.h"
#include <string>

START_NAMESPACE_STIR

/*!
\ingroup priors
\ingroup CUDA
\brief
A base class with CUDA-accelerated implementation of the Gibbsprior class.

This class provides CUDA-accelerated implementations of compute_value, compute_gradient,
accumulate_Hessian_times_input, compute_Hessian_diagonal and compute_gradient_times_input while inheriting all  parsing and
setting functionality from the base CPU GibbsPrior class.

Check the documentation of the GibbsPrior class for details on how the prior is defined and how to use it.

*/

template <typename elemT, typename PotentialT>
class CudaGibbsPrior : public GibbsPrior<elemT, PotentialT>
{
private:
  typedef GibbsPrior<elemT, PotentialT> base_type;

protected:
  // GPU block and grid dimensions

  dim3 block_dim;
  dim3 grid_dim;

  // Variable used for shared memory operations
  int threads_per_block = block_dim.x * block_dim.y * block_dim.z;
  size_t shared_mem_bytes = threads_per_block * sizeof(elemT);

  elemT* d_image_data = nullptr;
  // Currently stir:CartesianCoordinate3D<int> is not supported on GPU, we need a simple structure to store boundaries.
  int3 d_image_dim;
  int3 d_image_max_indices;
  int3 d_image_min_indices;
  int3 d_weight_max_indices;
  int3 d_weight_min_indices;

  // GPU pointers to weights and kappa data
  float* d_weights_data = nullptr;
  elemT* d_kappa_data = nullptr;

  // Buffers for GPU input/output to avoid reallocating memory on each call see usage in set_up() and ~CudaGibbsPrior()

  mutable double* d_scalar = nullptr;
  // d_scalar is used for compute_value and compute_gradient_times_input as output variable
  mutable elemT* d_input_data = nullptr;
  // d_input_data is used for storing input image for compute_gradient_times_input and accumulate_Hessian_times_input
  mutable elemT* d_output_data = nullptr;
  // d_output_data is used for storing output image for compute_gradient, accumulate_Hessian_times_input and
  // compute_Hessian_diagonal

public:
  CudaGibbsPrior();
  CudaGibbsPrior(const bool only_2D, float penalization_factor);
  ~CudaGibbsPrior();

  //! Override CPU version to set up CUDA resources on GPU and call parent set_up
  Succeeded set_up(shared_ptr<const DiscretisedDensity<3, elemT>> const& target_sptr) override;

  //! Set the weights array for the prior on the GPU.
  void set_weights(const Array<3, float>& w) override;

  //! Set the kappa image (spatially-varying penalty factors) on the GPU.
  void set_kappa_sptr(const shared_ptr<const DiscretisedDensity<3, elemT>>& k) override;

  //! CUDA-accelerated value computation
  double compute_value(const DiscretisedDensity<3, elemT>& current_image_estimate) override;

  //! CUDA-accelerated gradient computation
  void compute_gradient(DiscretisedDensity<3, elemT>& prior_gradient,
                        const DiscretisedDensity<3, elemT>& current_image_estimate) override;

  //! CUDA-accelerated gradient dot input computation
  double compute_gradient_times_input(const DiscretisedDensity<3, elemT>& input,
                                      const DiscretisedDensity<3, elemT>& current_image_estimate) override;

  //! CUDA-accelerated Hessian diagonal computation
  void compute_Hessian_diagonal(DiscretisedDensity<3, elemT>& Hessian_diagonal,
                                const DiscretisedDensity<3, elemT>& current_estimate) const override;

  //! CUDA-accelerated Hessian times input computation
  void accumulate_Hessian_times_input(DiscretisedDensity<3, elemT>& output,
                                      const DiscretisedDensity<3, elemT>& current_image_estimate,
                                      const DiscretisedDensity<3, elemT>& input) const override;
};

END_NAMESPACE_STIR

#ifdef __CUDACC__
// CUDA compiler sees everything
#  include "stir/recon_buildblock/CUDA/CudaGibbsPrior.cuh"
#endif

#endif // __stir_recon_buildblock_CUDA_CudaGibbsPrior_H__
