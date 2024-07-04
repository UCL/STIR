#include "stir/DiscretisedDensity.h"
#include "stir/recon_buildblock/GeneralisedPrior.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include "stir/IndexRange3D.h"
#include "stir/is_null_ptr.h"
#include "stir/Succeeded.h"

#include "stir/recon_buildblock/CUDA/CudaRelativeDifferencePrior.h"

extern "C" __global__
void computeCudaRelativeDifferencePriorGradientKernel(float* tmp_grad,
                                                    const float* image,
                                                    const float* weights,
                                                    const float* kappa,
                                                    const bool do_kappa,
                                                    const float gamma,
                                                    const float epsilon,
                                                    const float penalisation_factor,
                                                    const int z_dim,
                                                    const int y_dim,
                                                    const int x_dim) {
    // Get the voxel in x, y, z dimensions
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the voxel is within the image dimensions
    if (z >= z_dim || y >= y_dim || x >= x_dim) return;

    // Get the index of the voxel
    const int inputIndex = z * y_dim * x_dim + y * x_dim + x;

    // Define a single voxel gradient variable
    double voxel_gradient = 0.0f;

    // Define the neighbourhood
    int min_dz = -1;
    int max_dz = 1;
    int min_dy = -1;
    int max_dy = 1;
    int min_dx = -1;
    int max_dx = 1;
    
    // Check if the neighbourhood is at the boundary
    if (z == 0) min_dz = 0;
    if (z == z_dim - 1) max_dz = 0;
    if (y == 0) min_dy = 0;
    if (y == y_dim - 1) max_dy = 0;
    if (x == 0) min_dx = 0;
    if (x == x_dim - 1) max_dx = 0;

    // Apply RDP with hard coded 3x3x3 neighbourhood
    for(int dz = min_dz; dz <= max_dz; dz++) {
        for(int dy = min_dy; dy <= max_dy; dy++) {
            for(int dx = min_dx; dx <= max_dx; dx++) {
                const int neighbourIndex = (z + dz) * y_dim * x_dim + (y + dy) * x_dim + (x + dx);
                const int weightsIndex = (dz + 1) * 9 + (dy + 1) * 3 + (dx + 1);
                const float diff = (image[inputIndex] - image[neighbourIndex]);
                const float diff_abs = abs(diff);
                const float add = (image[inputIndex] + image[neighbourIndex]);
                const float add_3 = (image[inputIndex] + 3*image[neighbourIndex]);
                double current = weights[weightsIndex]*(diff*(gamma*diff_abs + add_3))/((add + gamma*diff_abs + epsilon)*(add + gamma*diff_abs + epsilon));
                if (do_kappa) {
                    current *= kappa[inputIndex]*kappa[neighbourIndex];
                }
                voxel_gradient += current;
            }
        }
    }
    tmp_grad[inputIndex] = penalisation_factor * voxel_gradient;
}


extern "C" __global__
void computeCudaRelativeDifferencePriorValueKernel(double* tmp_value,
                                                    const float* image,
                                                    const float* weights,
                                                    const float* kappa,
                                                    const bool do_kappa,
                                                    const float gamma,
                                                    const float epsilon,
                                                    const float penalisation_factor,
                                                    const int z_dim,
                                                    const int y_dim,
                                                    const int x_dim) {
    // Get the voxel in x, y, z dimensions
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the voxel is within the image dimensions
    if (z >= z_dim || y >= y_dim || x >= x_dim) return;

    // Get the index of the voxel
    const int inputIndex = z * y_dim * x_dim + y * x_dim + x;

    // Define the sum variable
    double sum = 0.0f;

    // Define the neighbourhood
    int min_dz = -1;
    int max_dz = 1;
    int min_dy = -1;
    int max_dy = 1;
    int min_dx = -1;
    int max_dx = 1;

    // Check if the neighbourhood is at the boundary
    if (z == 0) min_dz = 0;
    if (z == z_dim - 1) max_dz = 0;
    if (y == 0) min_dy = 0;
    if (y == y_dim - 1) max_dy = 0;
    if (x == 0) min_dx = 0;
    if (x == x_dim - 1) max_dx = 0;

    // Apply RDP with hard coded 3x3x3 neighbourhood
    for(int dz = min_dz; dz <= max_dz; dz++) {
        for(int dy = min_dy; dy <= max_dy; dy++) {
            for(int dx = min_dx; dx <= max_dx; dx++) {
                const int neighbourIndex = (z + dz) * y_dim * x_dim + (y + dy) * x_dim + (x + dx);
                const int weightsIndex = (dz + 1) * 9 + (dy + 1) * 3 + (dx + 1);

                const float diff = (image[inputIndex] - image[neighbourIndex]);
                const float add = (image[inputIndex] + image[neighbourIndex]);
                double current = (weights[weightsIndex]*0.5*diff*diff)/(add + gamma*abs(diff) + epsilon);
                if (do_kappa) {
                    current *= kappa[inputIndex]*kappa[neighbourIndex];
                }
                sum += current;
            }
        }
    }
    tmp_value[inputIndex] = penalisation_factor * sum;
}


START_NAMESPACE_STIR

//template <>
//const char* const CudaRelativeDifferencePrior<float>::registered_name = "Cuda Relative Difference Prior";
template class CudaRelativeDifferencePrior<float>;
//template <typename elemT>
//CudaRelativeDifferencePrior<elemT>::CudaRelativeDifferencePrior() : RelativeDifferencePrior<elemT>() {}
//template <typename elemT>
//CudaRelativeDifferencePrior<elemT>::CudaRelativeDifferencePrior(const bool only_2D, float penalization_factor, float gamma, float epsilon) : RelativeDifferencePrior<elemT>(only_2D, penalization_factor, gamma, epsilon) {}


END_NAMESPACE_STIR
