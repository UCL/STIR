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

  This class implements a Gibbs prior with relative difference potential function
  for regularization in image reconstruction.

  \author Matteo Neel Colombo
  \author Kris Thielemans
*/

#ifndef __stir_recon_buildblock_GibbsRelativeDifferencePrior_H__
#define __stir_recon_buildblock_GibbsRelativeDifferencePrior_H__


#include "stir/recon_buildblock/GibbsPrior.h"
#include "stir/RegisteredParsingObject.h"
#include <cmath>

#ifdef STIR_WITH_CUDA
#   include "stir/recon_buildblock/CUDA/CudaGibbsPrior.h"
#endif

START_NAMESPACE_STIR

/*!
  \ingroup priors
  \brief A Gibbs prior with Relative Difference Potential for image regularization.

  This class implements a Gibbs prior using a relative difference potential function,
  which is particularly effective for preserving edges while smoothing noise in 
  reconstructed images. The potential function normalizes differences by the sum
  of neighboring pixel values, making it adaptive to local image intensity.

  The prior energy is:
  \f[
  \sum_{r,dr} \kappa_{r} \kappa_{r+dr} w_{dr} V\left(\frac{f_r - f_{r+dr}}{f_r + f_{r+dr} + \gamma |f_r - f_{r+dr}| + \epsilon}\right)
  \f]
  where:
  - \f$f_r\f$ and \f$f_{r+dr}\f$ are pixel values at positions \f$r\f$ and \f$r+dr\f$
  - \f$w_{dr}\f$ are distance-dependent weights
  - \f$\kappa\f$ provides spatially-varying penalty weights (optional)
  - \f$\gamma\f$ controls edge preservation (higher values = more edge preservation)
  - \f$\epsilon\f$ prevents division by zero in uniform regions

  \par Usage Example:
  \code
  auto prior = std::make_shared<GibbsRelativeDifferencePrior<float>>();
  prior->set_penalisation_factor(0.01);
  prior->set_up(target_image);
  \endcode


  \see CudaGibbsRelativeDifferencePrior for CUDA-accelerated version
  \see GibbsQuadraticPrior for standard CPU version
*/


// Forward declaration for CudaRelativeDifferencePotential
#ifndef IGNORESWIG
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
  // const elemT add_3 = val_center + 3 * val_neigh;

  const elemT diff_abs = fabs(diff);
  const elemT A = add + gamma *diff_abs + epsilon;
  const elemT den = 1.0/ (A*A);
  
  // original
  // return 0.5 * (diff * (gamma * diff_abs + add_3 + 2 * epsilon)) / ((add + gamma * diff_abs + epsilon) * (add + gamma * diff_abs + epsilon));

  return 0.5 * (diff * (A + 2 * val_neigh +  epsilon))*den;

  }

  //! CUDA device function for computing the second derivative with respect to first argument
  __host__ __device__ inline double
  derivative_20(const elemT& val_center, const elemT& val_neigh, int z, int y, int x) const 
  {
    // elemT NUM =  (2* val_neigh + epsilon);
    // elemT DEN = (val_center + val_neigh + gamma * fabs(val_center - val_neigh) + epsilon);
    // return NUM*NUM/(DEN*DEN*DEN);
  return pow(2 * val_neigh + epsilon, 2) / pow(val_center + val_neigh + gamma * std::abs(val_center - val_neigh) + epsilon, 3);
  }
  //! CUDA device function for computing the mixed derivative
  __host__ __device__ inline double
  derivative_11(const elemT& val_center, const elemT& val_neigh, int z, int y, int x) const 
  {
  // For now, return 0 (not implemented)
  return - ((2*val_center  + epsilon) * (2*val_neigh + epsilon)) / pow(val_center + val_neigh + gamma * std::abs(val_center - val_neigh) + epsilon, 3);
  }
};
#endif // SWIG

template <typename elemT>
class RelativeDifferencePotential;

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

  void set_defaults() override;

protected:

  void initialise_keymap() override;

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

    void set_defaults() override;

  protected:
    
    void initialise_keymap() override;


  };
#endif

END_NAMESPACE_STIR

#endif
