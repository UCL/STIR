// Minimal CUDA TU to provide explicit template instantiations for Gibbs priors

/*
   SPDX-License-Identifier: Apache-2.0
*/

#include "stir/recon_buildblock/GibbsQuadraticPenalty.h"
#include "stir/recon_buildblock/GibbsRelativeDifferencePenalty.h"
#include "stir/recon_buildblock/CUDA/CudaGibbsPenalty.h"

START_NAMESPACE_STIR

// Ensure the CUDA base template is instantiated for the potentials we use
template class CudaGibbsPenalty<float, QuadraticPotential<float>>;
template class CudaGibbsPenalty<float, RelativeDifferencePotential<float>>;

END_NAMESPACE_STIR
