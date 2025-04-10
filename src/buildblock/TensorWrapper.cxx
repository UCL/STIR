#include "stir/TensorWrapper.h"

START_NAMESPACE_STIR


template <int num_dimensions, typename elemT>
TensorWrapper<num_dimensions, elemT>::TensorWrapper()
    : tensor(nullptr), device("cpu") {}

// Implementation of TensorWrapper methods
template <int num_dimensions, typename elemT>
TensorWrapper<num_dimensions, elemT>::TensorWrapper(const shared_ptr<at::Tensor> tensor, const std::string& device)
    : tensor((*tensor).to(device)), device(device) {}


template <int num_dimensions, typename elemT>
TensorWrapper<num_dimensions, elemT>::TensorWrapper(
    const IndexRange<num_dimensions> &  range, const std::string& device)
    : device(device)
{
    std::vector<int64_t> shape = convertIndexRangeToShape(range);
    torch::Dtype dtype = getTorchDtype();
    tensor = torch::zeros(shape, torch::TensorOptions().dtype(dtype));
    // Extract offsets
    offsets = extract_offsets_recursive(range);
    ends.resize(offsets.size());
    for(int i=0; i<offsets.size(); ++i)
      ends[i] = offsets[i] + get_length(i);
    tensor.to(device == "cuda" ? torch::kCUDA : torch::kCPU);
}

template <int num_dimensions, typename elemT>
TensorWrapper<num_dimensions, elemT>::TensorWrapper(
    const IndexRange<num_dimensions> &  range,
    shared_ptr<elemT[]> data_sptr,
    const std::string& device)
    : device(device)
{
  // tensor = torch::from_blob(data_sptr.get(), torch::TensorOptions().dtype(dtype));
  elemT* data_ptr = data_sptr.get();
  init(range, data_ptr);
  tensor.to(device == "cuda" ? torch::kCUDA : torch::kCPU);
}

template <int num_dimensions, typename elemT>
TensorWrapper<num_dimensions, elemT>::TensorWrapper(
    const shared_ptr<std::vector<int64_t>> shape,  const std::string& device)
    : device(device)
{
    if (shape->size() != num_dimensions) {
        throw std::invalid_argument("Shape size does not match the number of dimensions");
    }
    torch::Dtype dtype = getTorchDtype();
    tensor = torch::zeros((*shape), torch::TensorOptions().dtype(dtype));
    tensor.to(device == "cuda" ? torch::kCUDA : torch::kCPU);
}

// template <int num_dimensions, typename elemT>
// TensorWrapper<num_dimensions, elemT>::TensorWrapper(
//     const shared_ptr<stir::DiscretisedDensity<num_dimensions, elemT>> discretised_density,  const std::string& device)
//     : device(device)
// {
//     std::vector<int64_t> shape = getShapeFromDiscretisedDensity(discretised_density);
//     extract_offsets_recursive(discretised_density->get_index_range());
//     tensor = torch::zeros(shape, torch::TensorOptions().dtype(getTorchDtype()));
//     fillTensorFromDiscretisedDensity(discretised_density);
//     tensor.to(device == "cuda" ? torch::kCUDA : torch::kCPU);
// }

template <int num_dimensions, typename elemT>
void TensorWrapper<num_dimensions, elemT>::to_gpu()
{
    if (torch::cuda::is_available()) {
        device = "cuda";
        tensor = tensor.to(torch::kCUDA);
    } else {
        throw std::runtime_error("CUDA is not available. Cannot move tensor to GPU.");
    }
}

template <int num_dimensions, typename elemT>
void TensorWrapper<num_dimensions, elemT>::to_cpu() {
  device = "cpu";
    tensor = tensor.to(torch::kCPU);
}

template <int num_dimensions, typename elemT>
bool TensorWrapper<num_dimensions, elemT>::is_on_gpu() const {
    return tensor.device().is_cuda();
}

template <int num_dimensions, typename elemT>
void TensorWrapper<num_dimensions, elemT>::print_device() const {
    std::cout << "Tensor is on device: " << tensor.device() << std::endl;
}


template <int num_dimensions, typename elemT>
elemT TensorWrapper<num_dimensions, elemT>::find_max() const {
    return tensor.max().template item<elemT>();
}

template <int num_dimensions, typename elemT>
elemT TensorWrapper<num_dimensions, elemT>::sum() const {
    return tensor.sum().template item<elemT>();
}

template <int num_dimensions, typename elemT>
elemT TensorWrapper<num_dimensions, elemT>::sum_positive() const {
    return tensor.clamp_min(0).sum().template item<elemT>();
}

template <int num_dimensions, typename elemT>
elemT TensorWrapper<num_dimensions, elemT>::find_min() const {
    return tensor.min().template item<elemT>();
}

template <int num_dimensions, typename elemT>
void TensorWrapper<num_dimensions, elemT>::fill(const elemT& n) {
    tensor.fill_(n);
}

template <int num_dimensions, typename elemT>
void TensorWrapper<num_dimensions, elemT>::apply_lower_threshold(const elemT& l) {
    tensor.clamp_min(l);
}

template <int num_dimensions, typename elemT>
void TensorWrapper<num_dimensions, elemT>::apply_upper_threshold(const elemT& u) {
    tensor.clamp_max(u);
}

// template <int num_dimensions, typename elemT>
// TensorWrapper<num_dimensions, elemT> TensorWrapper<num_dimensions, elemT>::get_empty_copy() const {
//     torch::Tensor empty_tensor = torch::empty_like(tensor);
//     empty_tensor.to(device);
//     return TensorWrapper<num_dimensions, elemT>(empty_tensor);
// }


// template <int num_dimensions, typename elemT>
// std::unique_ptr<TensorWrapper<num_dimensions, elemT>> TensorWrapper<num_dimensions, elemT>::clone() const {
//     torch::Tensor cloned_tensor = tensor.clone();
//     tensor.to(device);
//     return std::make_unique<TensorWrapper<num_dimensions, elemT>>(cloned_tensor);
// }

template <int num_dimensions, typename elemT>
torch::Tensor& TensorWrapper<num_dimensions, elemT>::getTensor() {
    return tensor;
}

template <int num_dimensions, typename elemT>
const torch::Tensor& TensorWrapper<num_dimensions, elemT>::getTensor() const {
    return tensor;
}

template <int num_dimensions, typename elemT>
void TensorWrapper<num_dimensions, elemT>::printSizes() const {
    std::cout << "Tensor sizes: [";
    for (size_t i = 0; i < num_dimensions; ++i) {
        std::cout << tensor.size(i) << " ";
    }
    std::cout << "]" << std::endl;
}

template <int num_dimensions, typename elemT>
stir::IndexRange<num_dimensions> TensorWrapper<num_dimensions, elemT>::get_index_range() const {
    return stir::IndexRange<num_dimensions>();
}

template <int num_dimensions, typename elemT>
void TensorWrapper<num_dimensions, elemT>::resize(const stir::IndexRange<num_dimensions>& range) {
    std::vector<int64_t> new_shape = convertIndexRangeToShape(range);
    torch::Dtype dtype = getTorchDtype();
    tensor = torch::empty(new_shape, torch::TensorOptions().dtype(dtype));

    // Extract offsets
    offsets = extract_offsets_recursive(range);
    ends.resize(offsets.size());
    for(int i=0; i<offsets.size(); ++i)
      ends[i] = offsets[i] + get_length(i);
}

template <int num_dimensions, typename elemT>
void TensorWrapper<num_dimensions, elemT>::grow(const stir::IndexRange<num_dimensions>& range) {
    resize(range);
}

template <int num_dimensions, typename elemT>
void TensorWrapper<num_dimensions, elemT>::print() const {
    std::cout << "TensorWrapper Tensor:" << std::endl;
    std::cout << tensor << std::endl;
}

template <int num_dimensions, typename elemT>
void TensorWrapper<num_dimensions, elemT>::
    mask_cyl(const float radius, const bool in)
{
  float _radius = 0.0;
  if (radius == -1.f){
      _radius = static_cast<float>(get_max_index(2) - get_min_index(1)) / 2.f;
    }
  else
    _radius = radius;
  auto y_coords = torch::arange(0, tensor.size(1), tensor.options().dtype(torch::kFloat).device(tensor.device()));
  auto x_coords = torch::arange(0, tensor.size(2), tensor.options().dtype(torch::kFloat).device(tensor.device()));

  const float center_y = (tensor.size(1) - 1) / 2.0f;
  const float center_x = (tensor.size(2) - 1) / 2.0f;

  auto yy = y_coords.view({-1, 1}).expand({tensor.size(1), tensor.size(2)});
  auto xx = x_coords.view({1, -1}).expand({tensor.size(1), tensor.size(2)});

         // Compute the squared Euclidean distance in the x-y plane
  auto dist_squared = (yy - center_y).pow(2) + (xx - center_x).pow(2);
  auto mask_2d = dist_squared <= (_radius * _radius);
  auto mask_3d = mask_2d.unsqueeze(0).expand({tensor.size(0), tensor.size(1), tensor.size(2)});
  tensor.mul_(mask_3d);
}

// Explicit template instantiations
template class TensorWrapper<2, float>;
template class TensorWrapper<3, float>;
// template class TensorWrapper<4, float>;
END_NAMESPACE_STIR
