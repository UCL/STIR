#ifndef __stir_TENSOR_WRAPPER_H_
#define __stir_TENSOR_WRAPPER_H_

#include <vector>
#include <memory>
#include <stdexcept>
#include <torch/torch.h>
#include <iterator>

#include "stir/IndexRange.h"
#include "stir/shared_ptr.h"

START_NAMESPACE_STIR

           // Template class declaration for TensorWrapper
    template <int num_dimensions, typename elemT>
    class TensorWrapper {
private:
  typedef TensorWrapper<num_dimensions, elemT> self;
public:

  typedef elemT value_type;
  typedef elemT& reference;
  typedef const elemT& const_reference;
  typedef ptrdiff_t difference_type;
  typedef size_t size_type;

  typedef elemT full_value_type;
  typedef full_value_type* full_pointer;
  typedef const full_value_type* const_full_pointer;
  typedef full_value_type& full_reference;
  typedef const full_value_type& const_full_reference;

  // TensorAccessorIterator(torch::TensorAccessor<value_type, elemT> accessor, int index)
  //     : accessor_(accessor), index_(index) {}

         // Define full_iterator as a typedef for the tensor accessor
  typedef torch::TensorAccessor<elemT, num_dimensions> full_iterator;
  typedef torch::TensorAccessor<const elemT, num_dimensions> const_full_iterator;

  // // Example function to get the accessor
  // full_iterator get_accessor() {
  //   return tensor.accessor<elemT, num_dimensions>();
  // }

  //! Get const accessor
  const_full_iterator get_const_accessor() const {
    if (!tensor.is_contiguous()) {
        throw std::runtime_error("Tensor must be contiguous to use accessor.");
      }
    return tensor.accessor<const elemT, num_dimensions>();
  }

  TensorWrapper();
  TensorWrapper(const shared_ptr<torch::Tensor> tensor, const std::string& device = "cpu");
  explicit TensorWrapper(const IndexRange<num_dimensions> &range, const std::string& device = "cpu");
  TensorWrapper(const IndexRange<num_dimensions>& range, shared_ptr<elemT[]> data_sptr, const std::string& device = "cpu");

  TensorWrapper(const shared_ptr<std::vector<int64_t>> shape, const std::string& device = "cpu");
  // TensorWrapper(const shared_ptr<stir::DiscretisedDensity<num_dimensions, elemT> > discretised_density, const std::string& device = "cpu");

         //! Copy constructor
  inline TensorWrapper(const self& t)
      : tensor(t.tensor), device(t.device), offsets(t.offsets){}

  friend inline void swap(TensorWrapper& first, TensorWrapper& second) // nothrow
  {
    using std::swap;
    // Swap the member variables
    swap(first.tensor, second.tensor);
    swap(first.device, second.device);
    swap(first.offsets, second.offsets);
  }

  //! move constructor
  /*! implementation uses the copy-and-swap idiom, see e.g. https://stackoverflow.com/a/3279550 */
  inline TensorWrapper(TensorWrapper&& other) noexcept
  {
    swap(*this, other);
  }

  //! assignment operator
  /*! implementation uses the copy-and-swap idiom, see e.g. https://stackoverflow.com/a/3279550 */
  TensorWrapper& operator=(TensorWrapper other)
  {
    swap(*this, other);
    // info("Array= " + std::to_string(num_dimensions) + "copy of size " + std::to_string(this->size_all()));
    return *this;
  }

private:
  inline void init(const IndexRange<num_dimensions>& range, elemT* const data_ptr, bool copy_data = true)
  {
    // Determine the shape of the tensor
    std::vector<int64_t> shape = convertIndexRangeToShape(range);

    if (data_ptr == nullptr)
      {
        // Allocate a new tensor with the required shape
        tensor = torch::zeros(shape, torch::TensorOptions().dtype(getTorchDtype()));
      }
    else
      {
        tensor = torch::tensor(
            std::vector<elemT>(data_ptr, data_ptr + size_all()), // Copy data into a vector
            torch::TensorOptions().dtype(getTorchDtype())).reshape(shape); // Set dtype and shape
      }

           // Extract offsets
    offsets = extract_offsets_recursive(range);
  }

public:
  void to_gpu();
  void to_cpu();
  bool is_on_gpu() const;
  void print_device() const;

  inline int get_length() const {return static_cast<int>(size_all());}

  inline size_t size_all() const{return tensor.numel();}

  elemT find_max() const;

  elemT sum() const;

  elemT sum_positive() const;

  elemT find_min() const;

  void fill(const elemT& n);

  void apply_lower_threshold(const elemT& l);

  void apply_upper_threshold(const elemT& u);

  // TensorWrapper<num_dimensions, elemT> get_empty_copy() const;

  inline void _xapyb(const torch::Tensor& x, const elemT a, const torch::Tensor& y, const elemT b)
  {
    if (!x.sizes().equals(y.sizes())) {
        throw std::invalid_argument("Tensors x and y must have the same shape");
      }
    tensor = a * x + b * y;
  }

  inline void xapyb(const TensorWrapper<num_dimensions, elemT>& x, const elemT a, const TensorWrapper<num_dimensions, elemT>& y, const elemT b)
  {
    _xapyb(x.tensor, a, y.tensor, b);
  }

  inline void _xapyb(const torch::Tensor& x, const torch::Tensor& a, const torch::Tensor& y, const torch::Tensor& b)
  {
    if (!x.sizes().equals(a.sizes()) ||
        !x.sizes().equals(y.sizes()) ||
        !x.sizes().equals(b.sizes())) {
        throw std::invalid_argument("Tensors x and y must have the same shape");
      }
    tensor = a * x + b * y;
  }

  //! set values of the array to x*a+y*b, where a and b are arrays
  inline void xapyb(const TensorWrapper& x, const TensorWrapper& a, const TensorWrapper& y, const TensorWrapper& b)
  {
    _xapyb(x.tensor, a.tensor, y.tensor, b.tensor);
  }

  //        //! set values of the array to self*a+y*b where a and b are scalar or arrays
  template <class T>
  inline void sapyb(const T& a, const TensorWrapper& y, const T& b)
  {
    this->xapyb(*this, a, y, b);
  }

  // std::unique_ptr<TensorWrapper<num_dimensions, elemT>> clone() const;
  torch::Tensor& getTensor();
  const torch::Tensor& getTensor() const;
  void printSizes() const;

  stir::IndexRange<num_dimensions> get_index_range() const;

  void resize(const stir::IndexRange<num_dimensions>& range);

  void grow(const stir::IndexRange<num_dimensions>& range);

  inline bool is_contiguous() const{
    return tensor.is_contiguous();
  }
  void print() const;

protected:
  torch::Tensor tensor;
  std::string device; // Change from torch::Device to std::string
  std::vector<int> offsets;

  inline std::vector<int64_t> convertIndexRangeToShape(const stir::IndexRange<num_dimensions>& range) const
  {
    std::vector<int64_t> shape;
    computeShapeRecursive(range, shape);
    return shape;
  }

  template <int current_dim>
  inline void computeShapeRecursive(const stir::IndexRange<current_dim>& range, std::vector<int64_t>& shape) const
  {
    // Get the size of the current dimension
    shape.push_back(range.get_max_index() - range.get_min_index() + 1);
    if constexpr (current_dim > 1) {
        computeShapeRecursive(range[range.get_min_index()], shape);
      }
  }

         // std::vector<int64_t> getShapeFromDiscretisedDensity(const stir::shared_ptr<stir::Array<num_dimensions, elemT>> discretised_density) const
         // {
         //   const stir::IndexRange range = discretised_density->get_index_range();
         //   std::vector<int64_t> shape = convertIndexRangeToShape(range);
         //   return shape;
         // }

         // void fillTensorFromDiscretisedDensity(const stir::shared_ptr<stir::DiscretisedDensity<num_dimensions, elemT>> discretised_density)
         // {
         //   auto accessor = tensor.accessor<elemT, num_dimensions>();
         //   fillRecursive<num_dimensions>((*discretised_density),
         //                                 accessor,
         //                                 discretised_density->get_index_range());
         // }

         // template <int current_dim, typename Accessor>
         // inline void fillRecursive(const stir::Array<current_dim, elemT> & array,
         //                    Accessor accessor, const stir::IndexRange<current_dim>& range)
         // {
         //   // std::cout << "IN" << std::endl;
         //   if constexpr (current_dim == 1) {
         //       // std::cout << "DIM1" << std::endl;
         //       // Base case: Fill the last dimension
         //       int min_index = range.get_min_index();
         //       for (int i = range.get_min_index(); i < range.get_max_index(); ++i) {
         //           // std::cout << i << " " << i - range.get_min_index() << std::endl;
         //           accessor[i - min_index] = array[i];

         //         }
         //     } else {
         //       // Recursive case: Traverse the current dimension
         //       int min_index = range.get_min_index();
         //       // std::cout << "CUrrent dim" << current_dim << std::endl;
         //       for (int i = range.get_min_index(); i < range.get_max_index(); ++i) {
         //           // std::cout << "dc " << i << " " << i-range.get_min_index() << std::endl;
         //           fillRecursive<current_dim - 1>(array[i], accessor[i-min_index], range[i]);
         //         }
         //     }
         // }

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
  inline std::vector<int> extract_offsets_recursive(const stir::IndexRange<current_dim> &  range) {
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
