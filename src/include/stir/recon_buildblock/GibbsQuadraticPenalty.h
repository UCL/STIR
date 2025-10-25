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
  \brief Declaration of class stir::GibbsQuadraticPenalty, stir::CudaGibbsQuadraticPenalty
  and the potential function stir::QuadraticPotential

  \author Matteo Neel Colombo
  \author Kris Thielemans
*/

#ifndef __stir_recon_buildblock_GibbsQuadraticPenalty_H__
#define __stir_recon_buildblock_GibbsQuadraticPenalty_H__

#include "stir/RegisteredParsingObject.h"
#include "stir/cuda_utilities.h"
#include "stir/recon_buildblock/GibbsPenalty.h"
#ifdef STIR_WITH_CUDA
#  include "stir/recon_buildblock/CUDA/CudaGibbsPenalty.h"
#endif

START_NAMESPACE_STIR

#ifndef IGNORESWIG

/*!
  \ingroup priors
  \brief Implementation of the Quadratic penalty potential

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

  //! method to indicate whether the the prior defined by this potential is convex
  static inline bool is_convex() { return true; }
  //! Method for setting up parsing additional parameters
  void initialise_keymap(KeyParser& parser)
  {
    // No additional parameters needed for quadratic potential
  }
  //! Set default values for potential-specific parameters
  void set_defaults()
  {
    // No additional parameters for quadratic potential
  }
};

#else  // IGNORESWIG
template <typename elemT>
class QuadraticPotential;
#endif // SWIG

/*!
  \ingroup priors
  \brief Multithreaded CPU Implementation of the  Quadratic Gibbs prior

  \par Parsing
  These are the keywords that can be used in addition to the ones in GibbsPrior.
  \verbatim
  Gibbs Quadratic Penalty Parameters:=
  ; keywords from GibbsPrior
  END Gibbs Quadratic Penalty Parameters:=
  \endverbatim
*/
template <typename elemT>
class GibbsQuadraticPenalty : public RegisteredParsingObject<GibbsQuadraticPenalty<elemT>,
                                                             GeneralisedPrior<DiscretisedDensity<3, elemT>>,
                                                             GibbsPenalty<elemT, QuadraticPotential<elemT>>>
{
private:
  typedef RegisteredParsingObject<GibbsQuadraticPenalty<elemT>,
                                  GeneralisedPrior<DiscretisedDensity<3, elemT>>,
                                  GibbsPenalty<elemT, QuadraticPotential<elemT>>>
      base_type;

public:
  //! Name which will be used when parsing a GeneralisedPrior object
  static constexpr const char* const registered_name = "Gibbs Quadratic";

  GibbsQuadraticPenalty() { this->set_defaults(); }
  GibbsQuadraticPenalty(const bool only_2D, float penalisation_factor)
      : base_type(only_2D, penalisation_factor)
  {}
};

#ifdef STIR_WITH_CUDA
/*!
  \ingroup priors
  \brief GPU Implementation of the  Quadratic Gibbs prior

  \par Parsing
  These are the keywords that can be used in addition to the ones in GibbsPrior.
  \verbatim
  Cuda Gibbs Quadratic Prior Parameters:=
  ; keywords from GibbsPrior
  END Cuda Gibbs Quadratic Prior Parameters:=
  \endverbatim
*/
template <typename elemT>
class CudaGibbsQuadraticPenalty : public RegisteredParsingObject<CudaGibbsQuadraticPenalty<elemT>,
                                                                 GeneralisedPrior<DiscretisedDensity<3, elemT>>,
                                                                 CudaGibbsPenalty<elemT, QuadraticPotential<elemT>>>
{
private:
  typedef RegisteredParsingObject<CudaGibbsQuadraticPenalty<elemT>,
                                  GeneralisedPrior<DiscretisedDensity<3, elemT>>,
                                  CudaGibbsPenalty<elemT, QuadraticPotential<elemT>>>
      base_type;

public:
  //! Name which will be used when parsing a GeneralisedPrior object
  static constexpr const char* const registered_name = "Cuda Gibbs Quadratic";

  CudaGibbsQuadraticPenalty() { this->set_defaults(); }
  CudaGibbsQuadraticPenalty(const bool only_2D, float penalisation_factor)
      : base_type(only_2D, penalisation_factor)
  {}
};
#endif

END_NAMESPACE_STIR

#endif
