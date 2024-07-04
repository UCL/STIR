#ifndef __stir_recon_buildblock_CudaRelativeDifferencePrior_h__
#define __stir_recon_buildblock_CudaRelativeDifferencePrior_h__

#include "stir/recon_buildblock/GeneralisedPrior.h"
#include "stir/recon_buildblock/RelativeDifferencePrior.h"

START_NAMESPACE_STIR

// Define a native cpp struct to hold the dimensions of a CUDA grid or block
//struct cppdim3 {
//    int x;
//    int y;
//    int z;
//}; forward declaration instead

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
        using RelativeDifferencePrior<elemT>::RelativeDifferencePrior;
        // Name which will be used when parsing a GeneralisedPrior object
        inline static const char* const registered_name = "Cuda Relative Difference Prior";
        //static const char* const registered_name;
        //CudaRelativeDifferencePrior(const GeneralisedPrior<DiscretisedDensity<3, elemT>>& gp)
        //: RelativeDifferencePrior<elemT>(gp) {}
        // Constructors
        
        //! Default constructor
        CudaRelativeDifferencePrior();

        //! Constructs it explicitly
        CudaRelativeDifferencePrior(const bool only_2D, float penalization_factor, float gamma, float epsilon);

        //! Has to be called before using this object
        virtual Succeeded set_up(shared_ptr<DiscretisedDensity<3, elemT>> const& target_sptr);
        // Overridden methods
        //double compute_value(const DiscretisedDensity<3, elemT>& current_image_estimate) override;
        //void compute_gradient(DiscretisedDensity<3, elemT>& prior_gradient, const DiscretisedDensity<3, elemT>& current_image_estimate) override;

    //protected:
        //int z_dim, y_dim, x_dim;
        //cppdim3 cpp_grid_dim, cpp_block_dim;
        //bool cuda_set_up = false;
};

// Ensure the class is registered
//template class CudaRelativeDifferencePrior<float>;

END_NAMESPACE_STIR

#endif // __stir_recon_buildblock_CudaRelativeDifferencePrior_h__
