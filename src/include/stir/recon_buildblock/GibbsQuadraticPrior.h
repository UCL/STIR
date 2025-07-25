
#ifndef __stir_recon_buildblock_GibbsQuadraticPrior_H__
#define __stir_recon_buildblock_GibbsQuadraticPrior_H__


#include "stir/recon_buildblock/GibbsPrior.h"
#include "stir/RegisteredParsingObject.h"

#ifdef STIR_WITH_CUDA
# include "stir/recon_buildblock/CUDA/CudaGibbsPrior.h"
#endif

START_NAMESPACE_STIR



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
  };
#endif


END_NAMESPACE_STIR


#endif












