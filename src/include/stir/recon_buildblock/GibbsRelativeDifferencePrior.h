//
//
/*
    Copyright (C) 2025, University College London
    Copyright (C) 2025, University of Milano-Bicocca 
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup priors
  \ingroup CUDA
  \brief Declaration of class stir::CudaRelativeDifferencePrior, stir::CudaGibbsRelativeDifferencePrior and 
  the potential function stir::RelativeDifferencePotential

  \author Matteo Neel Colombo
  \author Kris Thielemans
*/

#ifndef __stir_recon_buildblock_GibbsRelativeDifferencePrior_H__
#define __stir_recon_buildblock_GibbsRelativeDifferencePrior_H__


#include "stir/recon_buildblock/GibbsPrior.h"
#include "stir/RegisteredParsingObject.h"
#include <cmath>
#include "stir/cuda_utilities.h"

#ifdef STIR_WITH_CUDA
#   include "stir/recon_buildblock/CUDA/CudaGibbsPrior.h"
#endif



START_NAMESPACE_STIR

#ifndef IGNORESWIG
 

/*!
  \ingroup priors
  \brief Implementation of the Relative Difference prior potential

  The prior energy is:
  \f[
  \sum_{r,dr} \kappa_{r} \kappa_{r+dr} w_{dr} \frac{(\lambda_r - \lambda_{r+dr})^2}{2(\lambda_r + \lambda_{r+dr} + \gamma |\lambda_r - \lambda_{r+dr}| + \epsilon)}
  \f]

  Where:
  - \f$\lambda_r\f$ and \f$\lambda_{r+dr}\f$ are pixel values at positions \f$r\f$ and \f$r+dr\f$
  - \f$\gamma\f$ is a smoothing parameter for the prior
  - \f$\epsilon\f$ prevents division by zero (default is 0)
  - \f$w_{dr}\f$ are distance-dependent weights
  - \f$\kappa\f$ provides spatially-varying penalty weights (optional)

  \warning Only well-defined for non-negative voxel values. With \f$\epsilon=0\f$, 
  gradient algorithms may have issues near zero values, as the Hessian becomes singular.
  
  \see Nuyts et al., "A Concave Prior Penalizing Relative Differences for Maximum-a-Posteriori 
  Reconstruction in Emission Tomography," IEEE TNS, vol. 49(1), pp. 56-60, 2002.
*/
template <typename elemT>
class RelativeDifferencePotential
{
public:
  float gamma;
  float epsilon;
  //! Method for computing the potential value
  __host__ __device__ inline double
  value(const elemT val_center, const elemT val_neigh, int z, int y, int x) const
  {
    // Implemented formula:
    // return 0.5 * (val_center -val_neigh)**2 / (val_center +val_neigh + gamma * |val_center - val_neigh| + epsilon); 
    const elemT diff = val_center - val_neigh;
    const elemT add  = val_center + val_neigh;
    const elemT NUM  = 0.5 * (diff * diff);
    const elemT DEN  = 1.0 / (add + gamma * fabs(diff) + epsilon);

    return NUM * DEN;
  }

  //! Method for computing the first derivative with respect to val_center
  __host__ __device__ inline double
  derivative_10(const elemT val_center, const elemT val_neigh, int z, int y, int x) const
  {   
  // Implemented formula:
  // return 0.5 * (val_center-val_neigh) * (val_center + 3*val_neigh + gamma * |val_center - val_neigh| + 2 * epsilon )  / ((val_center +val_neigh)+ gamma * |val_center - val_neigh| + epsilon)**2; 
  const elemT diff   = val_center - val_neigh;
  const elemT factor = val_center + val_neigh + gamma *fabs(diff) + epsilon;
  const elemT NUM    = 0.5 * diff * (factor + 2 * val_neigh +  epsilon);
  const elemT DEN    = 1.0/ (factor*factor);
  
  return NUM * DEN;
  }

  //! Method for computing the second derivative with respect to val_center
  __host__ __device__ inline double
  derivative_20(const elemT val_center, const elemT val_neigh, int z, int y, int x) const 
  {
    // Implemented formula:
    // return   (2*val_center + epsilon)**2 / (val_center + val_neigh + gamma * |val_center - val_neigh| + epsilon)**3; 
    const elemT NUM     =  2* val_neigh + epsilon;
    const elemT DEN     = 1.0/(val_center + val_neigh + gamma * fabs(val_center - val_neigh) + epsilon);

    return NUM*NUM * DEN*DEN*DEN;
  }
  //! Method for computing the mixed second derivative
  __host__ __device__ inline double
  derivative_11(const elemT val_center, const elemT val_neigh, int z, int y, int x) const 
  {
    // Implemented formula:
    // return   -(2*val_center + epsilon)*(2*val_neigh + epsilon)/ (val_center + val_neigh + gamma * |val_center - val_neigh| + epsilon)**3; 
    const elemT NUM = -(2*val_center  + epsilon) * (2*val_neigh  + epsilon);
    const elemT DEN = 1.0/(val_center + val_neigh + gamma * fabs(val_center - val_neigh) + epsilon);

    return  NUM * DEN*DEN*DEN;
  }
};
#endif // SWIG

template <typename elemT>
class RelativeDifferencePotential;

/*!
  \ingroup priors
  \brief CPU Implementation of the Relative Difference prior  
  \par Parsing
  These are the keywords that can be used in addition to the ones in GibbsPrior:
  \verbatim
  Gibbs Relative Difference Prior Parameters:= 
  ; next defaults to 0, set to 1 for 2D inverse Euclidean weights, 0 for 3D
  only 2D:= 0
  ; next can be used to set weights explicitly. Needs to be a 3D array (of floats).
  ; weights:={{{0,1,0},{1,0,1},{0,1,0}}}
  ; gamma value :=
  ; epsilon value :=
  ; use next parameter to specify an image with penalisation factors (a la Fessler)
  ; kappa filename:=
  ; use next parameter to get gradient images at every subiteration
  gradient filename prefix:=
  END Gibbs Relative Difference Prior Parameters:=
  \endverbatim
*/
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
  static constexpr const char* const registered_name = "Gibbs Relative Difference Prior";

  GibbsRelativeDifferencePrior();
  GibbsRelativeDifferencePrior(const bool only_2D, float penalisation_factor,float gamma_v, float epsilon_v);

  float get_gamma() const { return this->potential.gamma; }
  float get_epsilon() const { return this->potential.epsilon; }
  void  set_gamma(float gamma_v) { this->potential.gamma = gamma_v; }
  void  set_epsilon(float epsilon_v) { this->potential.epsilon = epsilon_v; }

  void set_defaults() override;
  bool is_convex() const override;

protected:

  void initialise_keymap() override;

};

#ifdef STIR_WITH_CUDA
  /*!
  \ingroup priors
  \brief GPU Implementation of the Relative Difference prior  
  \par Parsing
  These are the keywords that can be used in addition to the ones in CudaGibbsPrior:
  \verbatim
  Cuda Gibbs Relative Difference Prior Parameters:= 
  ; next defaults to 0, set to 1 for 2D inverse Euclidean weights, 0 for 3D
  only 2D:= 0
  ; next can be used to set weights explicitly. Needs to be a 3D array (of floats).
  ; weights:={{{0,1,0},{1,0,1},{0,1,0}}}
  ; gamma value :=
  ; epsilon value :=
  ; use next parameter to specify an image with penalisation factors (a la Fessler)
  ; kappa filename:=
  ; use next parameter to get gradient images at every subiteration
  gradient filename prefix:=
  END Cuda Gibbs Relative Difference Prior Parameters:=
  \endverbatim
  */
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
    static constexpr const char* const registered_name = "Cuda Gibbs Relative Difference Prior";

    CudaGibbsRelativeDifferencePrior();
    CudaGibbsRelativeDifferencePrior(const bool only_2D, float penalisation_factor,float gamma_v, float epsilon_v);

    float get_gamma() const { return this->potential.gamma; }
    float get_epsilon() const { return this->potential.epsilon; }
    void  set_gamma(float gamma_v) { this->potential.gamma = gamma_v; }
    void  set_epsilon(float epsilon_v) { this->potential.epsilon = epsilon_v; }

    void set_defaults() override;
    bool is_convex() const override;

  protected:
    
    void initialise_keymap() override;


  };
#endif

END_NAMESPACE_STIR

#endif
