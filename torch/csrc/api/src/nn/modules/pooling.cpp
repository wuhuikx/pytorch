#include <torch/nn/modules/pooling.h>

#include <torch/expanding_array.h>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

template <size_t D, typename Derived>
AvgPoolImpl<D, Derived>::AvgPoolImpl(const AvgPoolOptions<D>& options_)
    : options(options_) {}

template <size_t D, typename Derived>
void AvgPoolImpl<D, Derived>::reset() {}

template <size_t D, typename Derived>
void AvgPoolImpl<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::AvgPool" << D << "d"
         << "(kernel_size=" << options.kernel_size()
         << ", stride=" << options.stride()
         << ", padding=" << options.padding() << ")";
}

Tensor AvgPool1dImpl::forward(const Tensor& input) {
  return F::avg_pool1d(input, options);
}

Tensor AvgPool2dImpl::forward(const Tensor& input) {
  return F::avg_pool2d(input, options);
}

Tensor AvgPool3dImpl::forward(const Tensor& input) {
  return F::avg_pool3d(input, options);
}

template class AvgPoolImpl<1, AvgPool1dImpl>;
template class AvgPoolImpl<2, AvgPool2dImpl>;
template class AvgPoolImpl<3, AvgPool3dImpl>;

// ============================================================================

template <size_t D, typename Derived>
MaxPoolImpl<D, Derived>::MaxPoolImpl(const MaxPoolOptions<D>& options_)
    : options(options_) {}

template <size_t D, typename Derived>
void MaxPoolImpl<D, Derived>::reset() {}

template <size_t D, typename Derived>
void MaxPoolImpl<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::MaxPool" << D << "d"
         << "(kernel_size=" << options.kernel_size()
         << ", stride=" << options.stride()
         << ", padding=" << options.padding()
         << ", dilation=" << options.dilation()
         << ", ceil_mode=" << options.ceil_mode() << ")";
}

Tensor MaxPool1dImpl::forward(const Tensor& input) {
  return F::max_pool1d(input, options);
}

std::tuple<Tensor, Tensor> MaxPool1dImpl::forward_with_indices(const Tensor& input) {
  return F::max_pool1d_with_indices(input, options);
}

Tensor MaxPool2dImpl::forward(const Tensor& input) {
  return F::max_pool2d(input, options);
}

std::tuple<Tensor, Tensor> MaxPool2dImpl::forward_with_indices(const Tensor& input) {
  return F::max_pool2d_with_indices(input, options);
}

Tensor MaxPool3dImpl::forward(const Tensor& input) {
  return F::max_pool3d(input, options);
}

std::tuple<Tensor, Tensor> MaxPool3dImpl::forward_with_indices(const Tensor& input) {
  return F::max_pool3d_with_indices(input, options);
}

template class MaxPoolImpl<1, MaxPool1dImpl>;
template class MaxPoolImpl<2, MaxPool2dImpl>;
template class MaxPoolImpl<3, MaxPool3dImpl>;

} // namespace nn
} // namespace torch
