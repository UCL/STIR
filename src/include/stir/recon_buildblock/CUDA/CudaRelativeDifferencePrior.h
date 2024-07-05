#ifndef __stir_recon_buildblock_CudaRelativeDifferencePrior_h__
#define __stir_recon_buildblock_CudaRelativeDifferencePrior_h__

#include "stir/recon_buildblock/RelativeDifferencePrior.h"

START_NAMESPACE_STIR

struct cppdim3
{
  int x;
  int y;
  int z;
};

template <typename elemT>
class CudaRelativeDifferencePrior : public RegisteredParsingObject<CudaRelativeDifferencePrior<elemT>,
                                                                   GeneralisedPrior<DiscretisedDensity<3, elemT>>,
                                                                   RelativeDifferencePrior<elemT>>
{
private:
  typedef RegisteredParsingObject<CudaRelativeDifferencePrior<elemT>,
                                  GeneralisedPrior<DiscretisedDensity<3, elemT>>,
                                  RelativeDifferencePrior<elemT>>
      base_type;

public:
  // Name which will be used when parsing a GeneralisedPrior object
  inline static const char* const registered_name = "Cuda Relative Difference Prior";
#ifndef SWIG
  // import constructors from base_type
  // Note: currently disabled for SWIG as that needs SWIG 4.2
  using base_type::base_type;
#endif

  // Overridden methods
  double compute_value(const DiscretisedDensity<3, elemT>& current_image_estimate) override;
  void compute_gradient(DiscretisedDensity<3, elemT>& prior_gradient,
                        const DiscretisedDensity<3, elemT>& current_image_estimate) override;

  Succeeded set_up(shared_ptr<const DiscretisedDensity<3, elemT>> const& target_sptr) override;

protected:
  int z_dim, y_dim, x_dim;
  cppdim3 block_dim;
  cppdim3 grid_dim;
};

END_NAMESPACE_STIR

#endif // __stir_recon_buildblock_CudaRelativeDifferencePrior_h__
