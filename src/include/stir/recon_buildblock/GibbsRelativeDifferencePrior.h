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
  * the potential function stir::RelativeDifferencePotential

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
 
 /*!
  \file GibbsRelativeDifferencePrior.h
  \ingroup priors
  \brief
  A class in the GeneralisedPrior hierarchy. This implements a Relative Difference prior.

  The prior energy is:
  \f[
  \sum_{r,dr} \kappa_{r} \kappa_{r+dr} w_{dr} V\left(\frac{\lambda_r - \lambda_{r+dr}}{\lambda_r + \lambda_{r+dr} + \gamma |\lambda_r - \lambda_{r+dr}| + \epsilon}\right)
  \f]

  where \f$\lambda\f$ is the image and \f$r\f$ and \f$dr\f$ are indices and the sum
  is over the neighbourhood where the weights \f$w_{dr}\f$ are non-zero. \f$\gamma\f$ is
  a smoothing scalar term and the \f$\epsilon\f$ is a small non-negative value included to prevent division by zero.
  Please note that the RDP is only well defined for non-negative voxel values.
  For more details, see: <em> J. Nuyts, D. Beque, P. Dupont, and L. Mortelmans,
  "A Concave Prior Penalizing Relative Differences for Maximum-a-Posteriori Reconstruction in Emission Tomography,"
  vol. 49, no. 1, pp. 56-60, 2002. </em>

  If \f$ \epsilon=0 \f$, we attempt to resolve 0/0 at \f$ \lambda_r = \lambda_{r+dr}=0 \f$ by using the limit.
  Note that the Hessian explodes to infinity when both voxel values approach 0, and we currently return \c INFINITY.
  Also, as the RDP is not differentiable at this point, we have chosen to return 0 for the gradient
  (such that a zero background is not modified).

  \warning the default value for \f$ \epsilon \f$ is zero, which can be problematic for gradient-based algorithms.

  The \f$\kappa\f$ image can be used to have spatially-varying penalties such as in
  Jeff Fessler's papers. It should have identical dimensions to the image for which the
  penalty is computed. If \f$\kappa\f$ is not set, this class will effectively
  use 1 for all \f$\kappa\f$'s.

  By default, a 3x3 or 3x3x3 neighbourhood is used where the weights are set to
  x-voxel_size divided by the Euclidean distance between the points.

  The prior computation excludes voxel-pairs where one voxel is outside the volume. This is
  effectively the same as extending the volume by replicating the edges (which is different
  from zero boundary conditions).

\par Parsing
  These are the keywords that can be used in addition to the ones in GeneralPrior.
  \verbatim
  (Cuda) Gibbs Relative Difference Prior Parameters:= 
  ; next defaults to 0, set to 1 for 2D inverse Euclidean weights, 0 for 3D
  only 2D:= 0
  ; next can be used to set weights explicitly. Needs to be a 3D array (of floats).
  ' value of only_2D is ignored
  ; following example uses 2D 'nearest neighbour' penalty
  ; weights:={{{0,1,0},{1,0,1},{0,1,0}}}
  ; gamma value :=
  ; epsilon value :=
  ; see class documentation for more info
  ; use next parameter to specify an image with penalisation factors (a la Fessler)
  ; kappa filename:=
  ; use next parameter to get gradient images at every subiteration
  ; see class documentation
  gradient filename prefix:=
  END (Cuda) Gibbs Relative Difference Prior Parameters:=
  \endverbatim

  \see RelativeDifferencePrior for standard single core CPU version

*/

#ifndef IGNORESWIG
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
