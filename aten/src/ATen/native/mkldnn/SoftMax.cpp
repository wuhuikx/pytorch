#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

Tensor mkldnn_softmax(
    const Tensor& self,
    const int64_t dim,
    const bool half_to_float) {
  AT_ERROR("mkldnn_softmax: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_softmax_backward(
    const Tensor& grad_output,
    const Tensor& output,
    int64_t dim,
    const Tensor& self) {
  AT_ERROR("mkldnn_softmax_backward: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at {
namespace native {

Tensor mkldnn_softmax(
    const Tensor& self,
    const int64_t dim,
    const bool half_to_float) {
  AT_ASSERTM(
      !half_to_float,
      "softmax with half to float conversion is not supported on Mkldnn");
  const int64_t wrapped_dim = maybe_wrap_dim(dim, self.dim());
  ideep::tensor& x = itensor_from_mkldnn(self);
  ideep::tensor y;
  ideep::softmax_forward::compute(x, y, wrapped_dim);
  return new_with_itensor_mkldnn(std::move(y), self.options());
}

Tensor mkldnn_softmax_backward(
    const Tensor& grad_output,
    const Tensor& output,
    int64_t dim,
    const Tensor& self) {

  const int64_t wrapped_dim = maybe_wrap_dim(dim, self.dim());
  ideep::tensor& y = itensor_from_mkldnn(output);
  ideep::tensor& grady = itensor_from_mkldnn(grad_output);
  ideep::tensor gradx;
  ideep::softmax_backward::compute(y, grady, gradx, wrapped_dim);
  return new_with_itensor_mkldnn(std::move(gradx), self.options());
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
