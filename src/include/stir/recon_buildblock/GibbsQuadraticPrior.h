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
  \brief Declaration of class stir::GibbsQuadraticPrior, stir::CudaGibbsQuadraticPrior
  and the potential function stir::QuadraticPotential

  \author Matteo Neel Colombo
  \author Kris Thielemans
*/

#ifndef __stir_recon_buildblock_GibbsQuadraticPrior_H__
#define __stir_recon_buildblock_GibbsQuadraticPrior_H__

#include "stir/RegisteredParsingObject.h"
#include "stir/cuda_utilities.h"
#include "stir/recon_buildblock/GibbsPrior.h"
#ifdef STIR_WITH_CUDA
#  include "stir/recon_buildblock/CUDA/CudaGibbsPrior.h"
#endif

START_NAMESPACE_STIR

#ifndef IGNORESWIG

/*!
  \ingroup priors
  \brief Quadratic potential function for Gibbs priors

  The prior energy is:
  \f[
  \sum_{r,dr} \kappa_{r} \kappa_{r+dr} w_{dr} \frac{1}{2}(f_r - f_{r+dr})^2
  \f]
  where:
  - \f$f_r\f$ and \f$f_{r+dr}\f$ are pixel values at positions \f$r\f$ and \f$r+dr\f$
  - \f$w_{dr}\f$ are distance-dependent weights
  - \f$\kappa\f$ provides spatially-varying penalty weights (optional)
*/
template <typename elemT>
class QuadraticPotential
{
public:
  //! Method for computing the potential value
  __host__ __device__ inline double value(const elemT& val_center, const elemT val_neigh, int z, int y, int x) const
  {
    const elemT diff = val_center - val_neigh;
    return static_cast<double>(diff * diff) / 4.0;
  }
  //! Method for computing the first derivative with respect to val_center
  __host__ __device__ inline double derivative_10(const elemT val_center, const elemT val_neigh, int z, int y, int x) const
  {
    return static_cast<double>(val_center - val_neigh) / 2.0;
  }
  //! Method for computing the second derivative with respect to val_center
  __host__ __device__ inline double derivative_20(const elemT val_center, const elemT val_neigh, int z, int y, int x) const
  {
    return static_cast<double>(0.5);
  }
  //! Method for computing the mixed second derivative
  __host__ __device__ inline double derivative_11(const elemT val_center, const elemT val_neigh, int z, int y, int x) const
  {
    return static_cast<double>(-0.5);
  }
  //! Method for setting up parsing additional parameters
  void initialise_keymap(KeyParser& parser)
  {
    // No parameters needed for quadratic potential
  }
};
#endif // SWIG

template <typename elemT>
class QuadraticPotential;

/*!
  \ingroup priors
  \brief CPU Implementation of the  Quadratic Gibbs prior

  \par Parsing
  These are the keywords that can be used in addition to the ones in GibbsPrior.
  \verbatim
  Gibbs Quadratic Prior Parameters:=
  ; next defaults to 0, set to 1 for 2D inverse Euclidean weights, 0 for 3D
  only 2D:= 0
  ; next can be used to set weights explicitly. Needs to be a 3D array (of floats).
  ' value of only_2D is ignored
  ; following example uses 2D 'nearest neighbour' penalty
  ; weights:={{{0,1,0},{1,0,1},{0,1,0}}}
  ; use next parameter to specify an image with penalisation factors (a la Fessler)
  ; kappa filename:=
  ; use next parameter to get gradient images at every subiteration
  gradient filename prefix:=
  END Gibbs Quadratic Prior Parameters:=
  \endverbatim
*/
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
  static constexpr const char* const registered_name = "Gibbs Quadratic Prior";

  GibbsQuadraticPrior();
  GibbsQuadraticPrior(const bool only_2D, float penalisation_factor);

  void set_defaults() override;
  bool is_convex() const override;
  std::string get_parsing_name() const override;
};

#ifdef STIR_WITH_CUDA
/*!
  \ingroup priors
  \brief GPU Implementation of the  Quadratic Gibbs prior

  \par Parsing
  These are the keywords that can be used in addition to the ones in GibbsPrior.
  \verbatim
  Cuda Gibbs Quadratic Prior Parameters:=
  ; next defaults to 0, set to 1 for 2D inverse Euclidean weights, 0 for 3D
  only 2D:= 0
  ; next can be used to set weights explicitly. Needs to be a 3D array (of floats).
  ' value of only_2D is ignored
  ; following example uses 2D 'nearest neighbour' penalty
  ; weights:={{{0,1,0},{1,0,1},{0,1,0}}}
  ; use next parameter to specify an image with penalisation factors (a la Fessler)
  ; kappa filename:=
  ; use next parameter to get gradient images at every subiteration
  gradient filename prefix:=
  END Cuda Gibbs Quadratic Prior Parameters:=
  \endverbatim
*/
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
  static constexpr const char* const registered_name = "Cuda Gibbs Quadratic Prior";

  CudaGibbsQuadraticPrior();
  CudaGibbsQuadraticPrior(const bool only_2D, float penalisation_factor);

  void set_defaults() override;
  bool is_convex() const override;
  std::string get_parsing_name() const override;
};
#endif

END_NAMESPACE_STIR

#endif
