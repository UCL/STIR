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
  \brief Declaration of class stir::GibbsQuadraticPrior

  This class implements a Gibbs prior with quadratic potential function
  for regularization in image reconstruction.

  \author Matteo Neel Colombo
  \author Kris Thielemans
*/

#ifndef __stir_recon_buildblock_GibbsQuadraticPrior_H__
#define __stir_recon_buildblock_GibbsQuadraticPrior_H__


#include "stir/recon_buildblock/GibbsPrior.h"
#include "stir/RegisteredParsingObject.h"

#ifdef STIR_WITH_CUDA
#   include "stir/recon_buildblock/CUDA/CudaGibbsPrior.h"
#endif

START_NAMESPACE_STIR

/*!
  \ingroup priors
  \brief A Gibbs prior with Quadratic Potential for image regularization.

  This class implements a Gibbs prior using a quadratic potential function,
  which provides smooth regularization by penalizing differences between
  neighboring pixels. The quadratic potential is the simplest and most
  commonly used form of regularization in image reconstruction.

  The prior energy is:
  \f[
  \sum_{r,dr} \kappa_{r} \kappa_{r+dr} w_{dr} V\left(f_r - f_{r+dr}\right)
  \f]
  where:
  - \f$f_r\f$ and \f$f_{r+dr}\f$ are pixel values at positions \f$r\f$ and \f$r+dr\f$
  - \f$w_{dr}\f$ are distance-dependent weights
  - \f$\kappa\f$ provides spatially-varying penalty weights (optional)
  - \f$V(x) = \frac{1}{2}x^2\f$ is the quadratic potential function

    \par Usage Example:
  \code
  auto prior = std::make_shared<GibbsQuadraticPrior<float>>();
  prior->set_penalisation_factor(0.01);
  prior->set_up(target_image);
  \endcode


  \see CudaGibbsQuadraticPrior for CUDA-accelerated version
  \see GibbsQuadraticPrior for standard CPU version

*/



#ifndef IGNORESWIG
template <typename elemT>
class QuadraticPotential
{
public:

  //! CUDA device function for computing the potential value
  __host__ __device__ inline double
  value(const elemT& val_center, const elemT& val_neigh, int z, int y, int x) const
  {
    const elemT diff = val_center - val_neigh;
    return static_cast<double>(diff * diff) / 4.0;
  }
  //! CUDA device function for computing the first derivative with respect to first argument
  __host__ __device__ inline double
  derivative_10(const elemT& val_center, const elemT& val_neigh, int z, int y, int x) const 
  {
    return static_cast<double>(val_center - val_neigh) / 2.0;
  }
  //! CUDA device function for computing the second derivative with respect to first argument
  __host__ __device__ inline double
  derivative_20(const elemT& val_center, const elemT& val_neigh, int z, int y, int x) const 
  {
    return static_cast<double>(0.5);
  }
  //! CUDA device function for computing the mixed derivative
  __host__ __device__ inline double
  derivative_11(const elemT& val_center, const elemT& val_neigh, int z, int y, int x) const 
  {
    return static_cast<double>(-0.5);
  }
};
#endif // SWIG

template <typename elemT>
class QuadraticPotential;

// Device function implementations for CudaRelativeDifferencePotential
template <typename elemT>
class GibbsQuadraticPrior : public RegisteredParsingObject<GibbsQuadraticPrior<elemT>,
                                                            GeneralisedPrior<DiscretisedDensity<3, elemT>>,
                                                            GibbsPrior<elemT, QuadraticPotential<elemT>>>
{
private:
  typedef RegisteredParsingObject<GibbsQuadraticPrior<elemT>,
                                  GeneralisedPrior<DiscretisedDensity<3, elemT>>,
                                  GibbsPrior<elemT, QuadraticPotential<elemT>>>
      base_type;

public:
  //! Name which will be used when parsing a GeneralisedPrior object
  static constexpr const char* const registered_name = "GibbsQuadraticPrior";

  GibbsQuadraticPrior();
  GibbsQuadraticPrior(const bool only_2D, float penalisation_factor);

  void set_defaults() override;

protected:

  void initialise_keymap() override;

};



#ifdef STIR_WITH_CUDA
  template <typename elemT>
  class CudaGibbsQuadraticPrior : public RegisteredParsingObject<CudaGibbsQuadraticPrior<elemT>,
                                                                  GeneralisedPrior<DiscretisedDensity<3, elemT>>,
                                                                  CudaGibbsPrior<elemT, QuadraticPotential<elemT>>>
  {
  private:
    typedef RegisteredParsingObject<CudaGibbsQuadraticPrior<elemT>,
                                    GeneralisedPrior<DiscretisedDensity<3, elemT>>,
                                    CudaGibbsPrior<elemT, QuadraticPotential<elemT>>>
        base_type;

  public:
    //! Name which will be used when parsing a GeneralisedPrior object
    static constexpr const char* const registered_name = "CudaGibbsQuadraticPrior";

    CudaGibbsQuadraticPrior();
    CudaGibbsQuadraticPrior(const bool only_2D, float penalisation_factor);
    
    void set_defaults() override;

  protected:

    void initialise_keymap() override;

  };
#endif

END_NAMESPACE_STIR

#endif