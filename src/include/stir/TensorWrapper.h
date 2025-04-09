#ifndef __stir_TENSOR_WRAPPER_H_
#define __stir_TENSOR_WRAPPER_H_

#include <vector> // Ensure this is included
#include <memory>
#include <stdexcept>
#include <torch/torch.h>
#include "stir/DiscretisedDensity.h"
#include "stir/IndexRange.h"

START_NAMESPACE_STIR

// Template class declaration for TensorWrapper
template <int num_dimensions, typename elemT>
class TensorWrapper {
public:
  TensorWrapper(const torch::Tensor& tensor, const std::string& device = "cpu");
  TensorWrapper(const stir::IndexRange<num_dimensions>& range, const std::string& device = "cpu");
  TensorWrapper(const std::vector<int64_t>& shape, const std::string& device = "cpu");
  TensorWrapper(const stir::DiscretisedDensity<num_dimensions, elemT>& discretised_density, const std::string& device = "cpu");

    void to_gpu();
    void to_cpu();
    bool is_on_gpu() const;
    void print_device() const;
    size_t size_all() const;
    elemT find_max() const;
    elemT sum() const;
    elemT sum_positive() const;
    elemT find_min() const;
    void fill(const elemT& n);
    void apply_lower_threshold(const elemT& l);
    void apply_upper_threshold(const elemT& u);
    std::unique_ptr<TensorWrapper<num_dimensions, elemT>> get_empty_copy() const;
    void xapyb(const TensorWrapper<num_dimensions, elemT>& x, const elemT a, const TensorWrapper<num_dimensions, elemT>& y, const elemT b);
    std::unique_ptr<TensorWrapper<num_dimensions, elemT>> clone() const;
    torch::Tensor& getTensor();
    const torch::Tensor& getTensor() const;
    void printSizes() const;
    stir::IndexRange<num_dimensions> get_index_range() const;
    void resize(const stir::IndexRange<num_dimensions>& range);
    void grow(const stir::IndexRange<num_dimensions>& range);
    bool is_contiguous() const;
    void print() const;

protected:
    torch::Tensor tensor;
    std::string device; // Change from torch::Device to std::string
    std::vector<int> offsets;

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

private:
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
};

END_NAMESPACE_STIR

#endif // TENSOR_WRAPPER_H
