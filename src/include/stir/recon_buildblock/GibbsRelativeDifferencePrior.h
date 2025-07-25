//
//
/*
    Copyright (C) 2025, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup priors
  \ingroup CUDA
  \brief Declaration of class stir::CudaGibbsRelativeDifferencePrior

  \author Matteo Colombo
  \author Kris Thielemans
*/

#ifndef __stir_recon_buildblock_GibbsRelativeDifferencePrior_H__
#define __stir_recon_buildblock_GibbsRelativeDifferencePrior_H__


#include "stir/recon_buildblock/GibbsPrior.h"
#include "stir/RegisteredParsingObject.h"

#ifdef STIR_WITH_CUDA
#   include "stir/recon_buildblock/CUDA/CudaGibbsPrior.h"
#endif

START_NAMESPACE_STIR

/*!
  \ingroup priors
  \ingroup CUDA
  \brief A class in the GeneralisedPrior hierarchy.

  This implements a Gibbs prior with Relative Difference Potential.

  The prior is
  \f[
  \sum_{r,dr} \kappa_{r} \kappa_{r+dr} w_{dr} V((f_r - f_{r+dr})/(f_r + f_{r+dr} + \gamma |f_r - f_{r+dr}| + \epsilon))
  \f]
  where \f$w_{dr}\f$ are weights depending on the distance between \f$r\f$ and \f$r+dr\f$.
  The weights are computed such that the global maximum of the prior is independent
  of the voxel size.

  The \f$\kappa\f$ image can be used to have spatially-varying penalties such as in
  Jeff Fessler's papers. It should have identical dimensions to the image for which the
  penalty is computed. If \f$\kappa\f$ is not set, this prior will use 1 for all \f$\kappa_r\f$.

  By default, a 3x3x3 or 3x3 neighbourhood is used where the weights are set to
  x-voxel_size divided by the Euclidean distance between the points.

  \par Parameters for parsing

  \verbatim
  Gibbs Relative Difference Prior Parameters:=
  ; next defaults to 0, set to 1 to restrict the prior to 2D only
  only 2D := 0
  ; next can be used to set weights explicitly. Needs to be a 3D array (of floats).
  ; defaults to 3x3x3 or 3x3 as mentioned above
  ; weights := {{ {1,2,1}, {2,0,2}, {1,2,1} }}
  ; optional:
  ; kappa filename:=
  ; gamma := 1
  ; epsilon := 0
  END Gibbs Relative Difference Prior Parameters:=
  \endverbatim

*/


// Forward declaration for CudaRelativeDifferencePotential
template <typename elemT>
class RelativeDifferencePotential
{
public:
  float gamma;
  float epsilon;
  //! CUDA device function for computing the potential value
  __host__ __device__ inline double
  value(const elemT& val_center, const elemT& val_neigh, int z, int y, int x) const
  {
  const elemT diff = val_center - val_neigh;
  const elemT add = val_center + val_neigh;
  
  return 0.5 * (diff * diff) / (add + gamma * fabs(diff) + epsilon);
  }
  //! CUDA device function for computing the first derivative with respect to first argument
  __host__ __device__ inline double
  derivative_10(const elemT& val_center, const elemT& val_neigh, int z, int y, int x) const 
  {
  const elemT diff = val_center - val_neigh;
  const elemT add = val_center + val_neigh;
  const elemT diff_abs = fabs(diff);
  const elemT add_3 = val_center + 3 * val_neigh;

  return 0.5 * (diff * (gamma * diff_abs + add_3 + 2 * epsilon)) / ((add + gamma * diff_abs + epsilon) * (add + gamma * diff_abs + epsilon));
  }
  //! CUDA device function for computing the second derivative with respect to first argument
  __host__ __device__ inline double
  derivative_20(const elemT& val_center, const elemT& val_neigh, int z, int y, int x) const 
  {
  // For now, return 0 (not implemented)
  return 0.0;
  }
  //! CUDA device function for computing the mixed derivative
  __host__ __device__ inline double
  derivative_11(const elemT& val_center, const elemT& val_neigh, int z, int y, int x) const 
  {
  // For now, return 0 (not implemented)
  return 0.0;
  }
};

// Device function implementations for CudaRelativeDifferencePotential
template <typename elemT>
class GibbsRelativeDifferencePrior : public RegisteredParsingObject<GibbsRelativeDifferencePrior<elemT>,
                                                                    GeneralisedPrior<DiscretisedDensity<3, elemT>>,
                                                                    GibbsPrior<elemT,RelativeDifferencePotential<elemT>>>
{
private:
  typedef RegisteredParsingObject<GibbsRelativeDifferencePrior<elemT>,
                                  GeneralisedPrior<DiscretisedDensity<3, elemT>>,
                                  GibbsPrior<elemT, RelativeDifferencePotential<elemT>>>
      base_type;

public:
  //! Name which will be used when parsing a GeneralisedPrior object
  static constexpr const char* const registered_name = "GibbsRelativeDifferencePrior";

  GibbsRelativeDifferencePrior();
  GibbsRelativeDifferencePrior(const bool only_2D, float penalisation_factor,float gamma_v, float epsilon_v);

  float get_gamma() const { return this->potential.gamma; }
  float get_epsilon() const { return this->potential.epsilon; }
  void  set_gamma(float gamma_v) { this->potential.gamma = gamma_v; }
  void  set_epsilon(float epsilon_v) { this->potential.epsilon = epsilon_v; }

};


#ifdef STIR_WITH_CUDA
  template <typename elemT>
  class CudaGibbsRelativeDifferencePrior : public RegisteredParsingObject<CudaGibbsRelativeDifferencePrior<elemT>,
                                                                          GeneralisedPrior<DiscretisedDensity<3, elemT>>,
                                                                          CudaGibbsPrior<elemT, RelativeDifferencePotential<elemT>>>
  {
  private:
    typedef RegisteredParsingObject<CudaGibbsRelativeDifferencePrior<elemT>,
                                    GeneralisedPrior<DiscretisedDensity<3, elemT>>,
                                    CudaGibbsPrior<elemT, RelativeDifferencePotential<elemT>>>
        base_type;

  public:
    //! Name which will be used when parsing a GeneralisedPrior object
    static constexpr const char* const registered_name = "CudaGibbsRelativeDifferencePrior";

    CudaGibbsRelativeDifferencePrior();
    CudaGibbsRelativeDifferencePrior(const bool only_2D, float penalisation_factor,float gamma_v, float epsilon_v);

  float get_gamma() const { return this->potential.gamma; }
  float get_epsilon() const { return this->potential.epsilon; }
  void  set_gamma(float gamma_v) { this->potential.gamma = gamma_v; }
  void  set_epsilon(float epsilon_v) { this->potential.epsilon = epsilon_v; }

  };
#endif

END_NAMESPACE_STIR

#endif
