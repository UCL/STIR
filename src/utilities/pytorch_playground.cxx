//
//

/*!
\file
\ingroup utilities
\brief this executable is not meant to do something specific, other than facilitate the developlement of the Pytorch interface.
Heavily inspired by compare_images.cxxs

\author Nikos Efthimiou
*/


#include "stir/DiscretisedDensity.h"
#include "stir/ArrayFunction.h"
#include "stir/recon_array_functions.h"
#include "stir/IO/read_from_file.h"
#include "stir/is_null_ptr.h"
#include "stir/warning.h"
#include <numeric>
#include <stdlib.h>
#include <torch/torch.h>
#include <chrono> // For timing
#include <cuda_runtime.h>

using std::cerr;
using std::cout;
using std::endl;

//********************** main

template <int num_dimensions, typename elemT>
class TensorWrapper {
public:

  // Constructor: Initialize with a PyTorch tensor
  TensorWrapper(const torch::Tensor& tensor, torch::Device device = torch::kCPU)
      : tensor(tensor.to(device)), device(device) {}

  // Constructor: Initialize from an IndexRange
  TensorWrapper(const stir::IndexRange<num_dimensions>& range, torch::Device device = torch::kCPU)
      : device(device)  {
    // Convert IndexRange to a shape vector
    std::vector<int64_t> shape = convertIndexRangeToShape(range);
    // Determine the Torch data type based on elemT
    torch::Dtype dtype = getTorchDtype();

    // Extract offsets from the IndexRange and store them in the private vector
    offsets = extract_offsets_recursive(range);
    for(auto o : offsets){
        std::cout << "Offset " << o << std::endl;
      }

    // Create the tensor
    tensor = torch::zeros(shape, torch::TensorOptions().dtype(dtype));
    tensor.to(device);
  }

  // Constructor: Create an empty tensor with the specified shape
  TensorWrapper(const std::vector<int64_t>& shape, torch::Device device = torch::kCPU)
      : device(device)  {
    // Ensure the number of dimensions matches the shape size
    static_assert(num_dimensions > 0, "Number of dimensions must be greater than 0");
    if (shape.size() != num_dimensions) {
        throw std::invalid_argument("Shape size does not match the number of dimensions");
      }
    // Determine the Torch data type based on elemT
    torch::Dtype dtype = getTorchDtype();
    // Create the tensor
    tensor = torch::zeros(shape, torch::TensorOptions().dtype(dtype));
    tensor.to(device);
  }

    // Constructor: Initialize with a DiscretisedDensity
    TensorWrapper(const stir::DiscretisedDensity<num_dimensions, elemT>& discretised_density,
                torch::Device device = torch::kCPU) : device(device){
    // Get the shape from the DiscretisedDensity
    std::vector<int64_t> shape = getShapeFromDiscretisedDensity(discretised_density);
    // Extract offsets from the IndexRange and store them in the private vector
    offsets = extract_offsets_recursive(discretised_density.get_index_range());
    for(auto o : offsets){
        std::cout << "Offset " << o << std::endl;
      }
    // Create a PyTorch tensor with the same shape
    tensor = torch::zeros(shape, torch::TensorOptions().dtype(getTorchDtype()));
    // Fill the tensor with data from the DiscretisedDensity
    fillTensorFromDiscretisedDensity(discretised_density);
    tensor.to(device);
  }

    // Method to move the tensor to the GPU
    void to_gpu() {
    if (torch::cuda::is_available()) {
        device = torch::kCUDA;
        tensor = tensor.to(device);
      } else {
        throw std::runtime_error("CUDA is not available. Cannot move tensor to GPU.");
      }
  }

  // Method to move the tensor to the CPU
  void to_cpu() {
    device = torch::kCPU;
    tensor = tensor.to(device);
  }

  // Method to check if the tensor is on the GPU
  bool is_on_gpu() const {
    return tensor.device().is_cuda();
  }

  // Method to print the tensor's device
  void print_device() const {
    std::cout << "Tensor is on device: " << tensor.device() << std::endl;
  }

  size_t size_all() const{
    return tensor.numel();
  }

  elemT find_max() const{
    return tensor.max().template item<elemT>();
  }

  elemT sum() const{
    return tensor.sum().template item<elemT>();
  }

  elemT sum_positive() const{
    // This looks smart, if it works
    return tensor.clamp_min(0).sum().template item<elemT>();
  }

  elemT find_min() const{
    return tensor.min().template item<elemT>();
  }

  void fill(const elemT& n){
    tensor.fill_(n);
  }
  //! Sets elements below value to the value
  void apply_lower_threshold(const elemT& l){
    tensor.clamp_min(l);
  }

  //! Sets elements above value to the value
  void apply_upper_threshold(const elemT& u){
    tensor.clamp_max(u);
  }


  // Method to create an empty copy of the TensorWrapper
  std::unique_ptr<TensorWrapper<num_dimensions, elemT>> get_empty_copy() const {
    // Create an empty tensor with the same shape and data type as the current tensor
    torch::Tensor empty_tensor = torch::empty_like(tensor);
    // Probably dangerous, the User should move the data to the GPU when should
    empty_tensor.to(device);
    // Return a new TensorWrapper with the empty tensor
    return std::make_unique<TensorWrapper<num_dimensions, elemT>>(empty_tensor);
  }

  // Method to perform xapyb: result = a * x + b * y
  void xapyb(
      const TensorWrapper<num_dimensions, elemT>& x, const elemT a,
      const TensorWrapper<num_dimensions, elemT>& y, const elemT b) {

    // Ensure x and y have the same shape
    if (!x.tensor.sizes().equals(y.tensor.sizes())) {
        throw std::invalid_argument("Tensors x and y must have the same shape");
      }
        // Perform the operation: result = a * x + b * y
    tensor = a * x.tensor + b * y.tensor;
  }


  // Method to clone the TensorWrapper
  std::unique_ptr<TensorWrapper<num_dimensions, elemT>> clone() const {
    // Create a deep copy of the underlying tensor
    torch::Tensor cloned_tensor = tensor.clone();
     // Probably dangerous, the User should move the data to the GPU when should
    tensor.to(device);
    // Return a new TensorWrapper with the cloned tensor
    return std::make_unique<TensorWrapper<num_dimensions, elemT>>(cloned_tensor);
  }



         // // Overload operator[] for the first dimension
         // auto operator[](int64_t index) {
         //   // Return a TensorWrapper for the next dimension
         //   return TensorWrapper<num_dimensions - 1, elemT>(tensor[index]);
         // }

         //        // Overload operator[] for the last dimension
         // elemT& operator[](int64_t index) requires(num_dimensions == 1) {
         //   return tensor.data_ptr<elemT>()[index];
         // }


         // Access the underlying tensor
  torch::Tensor& getTensor() {
    return tensor;
  }

  const torch::Tensor& getTensor() const {
    return tensor;
  }

         // Method to print the sizes of the tensor
  void printSizes() const {
    std::cout << "Tensor sizes: [";
    for (size_t i = 0; i < num_dimensions; ++i) {
        std::cout << tensor.size(i) << " ";
      }
    std::cout << "]" << std::endl;
  }

  stir::IndexRange<num_dimensions> get_index_range() const {
    return stir::IndexRange<num_dimensions>();
  }

  void resize(const stir::IndexRange<num_dimensions>& range)
  {
    // Convert IndexRange to a shape vector
    std::vector<int64_t> new_shape = convertIndexRangeToShape(range);

           // Determine the Torch data type based on elemT
    torch::Dtype dtype = getTorchDtype();

           // Create the tensor
    tensor = torch::empty(new_shape, torch::TensorOptions().dtype(dtype));
  }

  //! alias for resize()
  virtual inline void grow(const stir::IndexRange<num_dimensions>& range)
  {
    resize(range);
  }

  bool is_contiguous() const{
    return tensor.is_contiguous();
  }

  // Print the tensor
  void print() const {
    std::cout << "TensorWrapper Tensor:" << std::endl;
    std::cout << tensor << std::endl;
  }

protected:
  // Well, the actual container.
  torch::Tensor tensor;
  torch::Device device; // The device (CPU or GPU) where the tensor is stored
  std::vector<int> offsets;          // Offsets for each dimension

         // Helper function to map IndexRange to a shape vector
         // template <int num_dimensions>
  std::vector<int64_t> convertIndexRangeToShape(const stir::IndexRange<num_dimensions>& range) const {
    std::vector<int64_t> shape;
    computeShapeRecursive(range, shape);
    return shape;
  }

  template <int current_dim>
  void computeShapeRecursive(const stir::IndexRange<current_dim>& range, std::vector<int64_t>& shape) const {
    // Get the size of the current dimension
    shape.push_back(range.get_max_index() - range.get_min_index() + 1);

           // Recurse for the next dimension if it exists
    if constexpr (current_dim > 1) {
        computeShapeRecursive(range[range.get_min_index()], shape);
      }
  }

  // Helper function to get the shape from a DiscretisedDensity
  std::vector<int64_t> getShapeFromDiscretisedDensity(const stir::Array<num_dimensions, elemT>& discretised_density) const {
    const stir::IndexRange range = discretised_density.get_index_range();
    std::vector<int64_t> shape = convertIndexRangeToShape(range);
    // std::cout<< shape[0] << shape[1] << shape[2] << std::endl;
    return shape;
  }

  void fillTensorFromDiscretisedDensity(const stir::DiscretisedDensity<num_dimensions, elemT>& discretised_density) {
    auto accessor = tensor.accessor<elemT, num_dimensions>();
    fillRecursive<num_dimensions>(discretised_density, accessor, discretised_density.get_index_range());
  }

  template <int current_dim, typename Accessor>
  void fillRecursive(const stir::Array<current_dim, elemT>& array,
                     Accessor accessor, const stir::IndexRange<current_dim>& range) {
    // std::cout << "IN" << std::endl;
    if constexpr (current_dim == 1) {
        // std::cout << "DIM1" << std::endl;
        // Base case: Fill the last dimension
        int min_index = range.get_min_index();
        for (int i = range.get_min_index(); i < range.get_max_index(); ++i) {
            // std::cout << i << " " << i - range.get_min_index() << std::endl;
            accessor[i - min_index] = array[i];

          }
      } else {
        // Recursive case: Traverse the current dimension
        int min_index = range.get_min_index();
        // std::cout << "CUrrent dim" << current_dim << std::endl;
        for (int i = range.get_min_index(); i < range.get_max_index(); ++i) {
            // std::cout << "dc " << i << " " << i-range.get_min_index() << std::endl;
            fillRecursive<current_dim - 1>(array[i], accessor[i-min_index], range[i]);
          }
      }
  }

  template <int current_dim>
  std::vector<int> extract_offsets_recursive(const stir::IndexRange<current_dim>& range) {
    std::vector<int> result;
    result.push_back(range.get_min_index()); // Get the minimum index for the current dimension

    if constexpr (current_dim > 1) {
        // Recurse into the next dimension
        auto sub_offsets = extract_offsets_recursive(range[range.get_min_index()]);
        result.insert(result.end(), sub_offsets.begin(), sub_offsets.end());
      }

    return result;
  }

private:
  // Helper function to map elemT to Torch data type
  torch::Dtype getTorchDtype() const {
    if constexpr (std::is_same<elemT, float>::value) {
        return torch::kFloat;
      } else if constexpr (std::is_same<elemT, double>::value) {
        return torch::kDouble;
      } else if constexpr (std::is_same<elemT, int>::value) {
        return torch::kInt;
      } else {
        throw std::invalid_argument("Unsupported data type");
      }
  }
};


USING_NAMESPACE_STIR



int
main(int argc, char* argv[])
{
  if (argc < 3 || argc > 7)
    {
      cerr << "Usage: \n"
           << argv[0] << "\n\t"
           << "[-r rimsize] \n\t"
           << "[-t tolerance] \n\t"
           << "old_image new_image \n\t"
           << "'rimsize' has to be a nonnegative integer.\n\t"
           << "'tolerance' is by default .0005 \n\t"
           << "When the -r option is used, the (radial) rim of the\n\t"
           << "images will be set to 0, for 'rimsize' pixels.\n";
      return (EXIT_FAILURE);
    }
  // skip program name
  --argc;
  ++argv;
  int rim_truncation_image = -1;
  float tolerance = .0005F;

  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);

  if (err != cudaSuccess) {
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
      return -1;
    }

  if (device_count == 0) {
      std::cout << "No CUDA devices available." << std::endl;
      return 0;
    }

  std::cout << "Number of CUDA devices: " << device_count << std::endl;

  for (int device = 0; device < device_count; ++device) {
      cudaDeviceProp device_prop;
      cudaGetDeviceProperties(&device_prop, device);

      std::cout << "Device " << device << ": " << device_prop.name << std::endl;
      std::cout << "  CUDA Capability: " << device_prop.major << "." << device_prop.minor << std::endl;
      std::cout << "  Total Memory: " << device_prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
      std::cout << "  Multiprocessors: " << device_prop.multiProcessorCount << std::endl;
      std::cout << "  Clock Rate: " << device_prop.clockRate / 1000 << " MHz" << std::endl;
    }

  // first process command line options
  while (argc > 0 && argv[0][0] == '-')
    {
      if (strcmp(argv[0], "-r") == 0)
        {
          if (argc < 2)
            {
              cerr << "Option '-r' expects a nonnegative (integer) argument\n";
              exit(EXIT_FAILURE);
            }
          rim_truncation_image = atoi(argv[1]);
          argc -= 2;
          argv += 2;
        }
      if (strcmp(argv[0], "-t") == 0)
        {
          if (argc < 2)
            {
              cerr << "Option '-t' expects a (float) argument\n";
              exit(EXIT_FAILURE);
            }
          tolerance = static_cast<float>(atof(argv[1]));
          argc -= 2;
          argv += 2;
        }
    }

  shared_ptr<DiscretisedDensity<3, float>> first_operand(read_from_file<DiscretisedDensity<3, float>>(argv[0]));

  if (is_null_ptr(first_operand))
    {
      cerr << "Could not read first file\n";
      exit(EXIT_FAILURE);
    }

  stir::IndexRange d = first_operand->get_index_range();

  std::cout << "D:" << d.get_min_index() << " " << d.get_max_index() << " " << d.get_length() << std::endl;
  std::cout << "D:" << d[0].get_min_index() << " " << d[0].get_max_index() << " " << d[0].get_length() << std::endl;
  std::cout << "G:" << d[0][0].get_min_index() << " " << d[0][0].get_max_index() << " " << d[0][0].get_length() << std::endl;

  std::vector<int64_t> shape({d.get_length(), d[0].get_length(), d[0][0].get_length()});
  // Allocate Tensor
  // TensorWrapper<3, float> tw(shape);
  TensorWrapper tw((*first_operand));

  // Print the tensor's device
  tw.print_device();
  // Move the tensor to the GPU (if available)
  try {
      tw.to_gpu();
      std::cout << "Moved tw to GPU." << std::endl;
    } catch (const std::runtime_error& e) {
      std::cerr << e.what() << std::endl;
    }
  // Print the tensor's device again
  tw.print_device();

  std::cout <<  " H " << std::endl;
  tw.printSizes();
  std::cout <<  " H2 " << std::endl;

  const float a = 1000;
  const float b = 5000;
  // Start timing
  {
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "stir::Array MAX value: " << (*first_operand).find_max() << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "stir::Array: Time to compute max value: " << duration << " ms" << std::endl;
    std::cout << "\n" << std::endl;
  }
  {
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Tensored: MAX value: " << tw.find_max() << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Tensored: Time to compute max value: " << duration << " ms" << std::endl;
    std::cout << "\n" << std::endl;
  }

  {
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "stir::Array SUM value: " << (*first_operand).sum() << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "stir::Array: Time to compute SUM value: " << duration << " ms" << std::endl;
    std::cout << "\n" << std::endl;
  }
  {
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Tensored: SUM value: " << tw.sum() << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Tensored: Time to compute SUM value: " << duration << " ms" << std::endl;
    std::cout << "\n" << std::endl;
  }

  {
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "stir::Array SUM_pos value: " << (*first_operand).sum_positive() << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "stir::Array: Time to compute SUM_pos value: " << duration << " ms" << std::endl;
    std::cout << "\n" << std::endl;
  }
  {
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Tensored: SUM_pos value: " << tw.sum_positive() << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Tensored: Time to compute SUM_pos value: " << duration << " ms" << std::endl;
    std::cout << "\n" << std::endl;
  }

  std::cout << "____________XAPYB__________" << std::endl;
  {
    auto cloned_empty_first_operand = first_operand->get_empty_copy();
    auto cloned_first_operand = first_operand->clone();
    auto start = std::chrono::high_resolution_clock::now();
    cloned_empty_first_operand->xapyb((*first_operand), a, *cloned_first_operand, b);
    std::cout << "stir::Array: MAX value after xapyb: " << cloned_empty_first_operand->find_max() << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "stir::Array: Time to compute max value after xapyb: : " << duration << " ms" << std::endl;
    std::cout << "\n" << std::endl;
  }

  {
    auto cloned_empty_tw = tw.get_empty_copy();
    cloned_empty_tw->print_device();
    try {
        cloned_empty_tw->to_gpu();
        std::cout << "Moved cloned_empty_tw to GPU." << std::endl;
      } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
      }
    // Print the tensor's device again
    cloned_empty_tw->print_device();

    auto cloned_tw = tw.clone();
    cloned_tw->print_device();
    try {
        cloned_tw->to_gpu();
        std::cout << "Moved cloned_tw to GPU." << std::endl;
      } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
      }
    // Print the tensor's device again
    cloned_tw->print_device();

    auto start = std::chrono::high_resolution_clock::now();
    cloned_empty_tw->xapyb(tw, a, *cloned_tw, b);
    std::cout << "Tensored: MAX value after xapyb: " << cloned_empty_tw->find_max() << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Tensored: Time to compute max value: " << duration << " ms" << std::endl;
    std::cout << "\n" << std::endl;
  }

         // {
         //   auto start = std::chrono::high_resolution_clock::now();
         //   std::cout << "Tensored first_operand MAX value: " << tw.find_max() << std::endl;
         //   auto end = std::chrono::high_resolution_clock::now();
         //   auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
         //   std::cout << "Time to compute max value: " << duration << " ms" << std::endl;
         // }


  std::cout << "STOP HERE" << std::endl;
  return 1;

  // shared_ptr<DiscretisedDensity<3, float>> second_operand(read_from_file<DiscretisedDensity<3, float>>(argv[1]));
  // if (is_null_ptr(second_operand))
  //   {
  //     cerr << "Could not read 2nd file\n";
  //     exit(EXIT_FAILURE);
  //   }

  //        // check if images are compatible
  // {
  //   std::string explanation;
  //   if (!first_operand->has_same_characteristics(*second_operand, explanation))
  //     {
  //       warning("input images do not have the same characteristics.\n%s", explanation.c_str());
  //       return EXIT_FAILURE;
  //     }
  // }

  // if (rim_truncation_image >= 0)
  //   {
  //     truncate_rim(*first_operand, rim_truncation_image);
  //     truncate_rim(*second_operand, rim_truncation_image);
  //   }

  // float reference_max = first_operand->find_max();
  // float reference_min = first_operand->find_min();

  // float amplitude = fabs(reference_max) > fabs(reference_min) ? fabs(reference_max) : fabs(reference_min);

  // *first_operand -= *second_operand;
  // const float max_error = first_operand->find_max();
  // const float min_error = first_operand->find_min();
  // in_place_abs(*first_operand);
  // const float max_abs_error = first_operand->find_max();

  // const bool same = (max_abs_error / amplitude <= tolerance);

  // cout << "\nMaximum absolute error = " << max_abs_error << "\nMaximum in (1st - 2nd) = " << max_error
  //      << "\nMinimum in (1st - 2nd) = " << min_error << endl;
  // cout << "Error relative to sup-norm of first image = " << (max_abs_error / amplitude) * 100 << " %" << endl;

  // cout << "\nImage arrays ";

  // if (same)
  //   {
  //     cout << (max_abs_error == 0 ? "are " : "deemed ") << "identical\n";
  //   }
  // else
  //   {
  //     cout << "deemed different\n";
  //   }
  // cout << "(tolerance used: " << tolerance * 100 << " %)\n\n";
  // return same ? EXIT_SUCCESS : EXIT_FAILURE;

} // end main
