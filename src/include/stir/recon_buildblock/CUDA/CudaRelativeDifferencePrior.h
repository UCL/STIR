#ifndef __stir_recon_buildblock_CudaRelativeDifferencePrior_h__
#define __stir_recon_buildblock_CudaRelativeDifferencePrior_h__

#include "stir/recon_buildblock/RelativeDifferencePrior.h"
#include "stir/DiscretisedDensity.h"
#include <cuda_runtime.h>
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

template <typename elemT>
class CudaRelativeDifferencePrior : public RelativeDifferencePrior<elemT> {
    public:
        using RelativeDifferencePrior<elemT>::RelativeDifferencePrior;
        // Name which will be used when parsing a GeneralisedPrior object
        static const char* const registered_name;

        // Constructors
        CudaRelativeDifferencePrior();
        CudaRelativeDifferencePrior(const bool only_2D, float penalization_factor, float gamma, float epsilon);

        // Overridden methods
        double compute_value(const DiscretisedDensity<3, elemT>& current_image_estimate) override;
        void compute_gradient(DiscretisedDensity<3, elemT>& prior_gradient, const DiscretisedDensity<3, elemT>& current_image_estimate) override;

        // New methods
        Succeeded set_up_cuda(shared_ptr<DiscretisedDensity<3, elemT>> const& target_sptr);
    protected:
        int z_dim, y_dim, x_dim;
        dim3 grid_dim, block_dim;
        bool cuda_set_up = false;
};

END_NAMESPACE_STIR

#endif // __stir_recon_buildblock_CudaRelativeDifferencePrior_h__