#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

Tensor mkldnn_linear(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias) {
  TORCH_CHECK(false, "mkldnn_linear: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_linear_backward_input(
    IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight) {
  TORCH_CHECK(false, "mkldnn_linear_backward_input: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor> mkldnn_linear_backward_weights(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight, bool bias_defined) {
  TORCH_CHECK(false, "mkldnn_linear_backward_weights: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_linear_backward(
    const Tensor& input, const Tensor& grad_output_t,
    const Tensor& weight, std::array<bool,3> output_mask) {
  TORCH_CHECK(false, "mkldnn_linear_backward: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at {
namespace native {

Tensor mkldnn_linear(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias) {
  TORCH_CHECK(self.dim() == 2 || self.dim() == 3,
      "mkldnn_linear: only support input with 2 or 3 dim, input dim ", self.dim());
  TORCH_CHECK(self.is_mkldnn(),
      "mkldnn_linear: input needs to be mkldnn layout");

  const ideep::tensor x = itensor_from_mkldnn(self);
  const ideep::tensor w_trans = itensor_from_tensor(weight).transpose_(0, 1);
  
  auto input_size = self.sizes();
  std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
  output_size.push_back(weight.size(0));

  ideep::tensor y;
  if (bias.defined()) {
    ideep::tensor b = itensor_from_tensor(bias);
    ideep::tensor::dims bias_dims(output_size.size() - 1, 1);
    bias_dims.push_back(output_size.back());
    auto bia_tag = bias_dims.size() == 2 ? ideep::format_tag::ab : ideep::format_tag::abc;
    b.set_desc({bias_dims, b.get_data_type(), bia_tag});
    ideep::matmul_forward::compute(x, w_trans, b, y);
  } else {
    ideep::matmul_forward::compute(x, w_trans, y);
  }

  if (self.dim() > 2) {
    return new_with_itensor_mkldnn(std::move(y), self.options()).reshape(output_size);
  }
  return new_with_itensor_mkldnn(std::move(y), self.options());
}

Tensor mkldnn_linear_backward_input(
    IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight){
  auto grad_output_reshaped = grad_output.dim() > 2 ?
    grad_output.reshape({-1, grad_output.size(grad_output.dim() - 1)}) : grad_output;
  ideep::tensor& grady = itensor_from_mkldnn(grad_output_reshaped);
  const ideep::tensor w = itensor_from_tensor(weight);

  std::vector<int64_t> input_reshaped_size;
  input_reshaped_size.push_back(grad_output_reshaped.size(0));
  input_reshaped_size.push_back(weight.size(1));

  ideep::tensor gradx;
  ideep::inner_product_backward_data::compute(
    grady, w, {input_reshaped_size.begin(), input_reshaped_size.end()}, gradx);

  if (input_size.size() > 2) {
    return new_with_itensor_mkldnn(std::move(gradx), grad_output.options()).reshape(input_size);
  }
  return new_with_itensor_mkldnn(std::move(gradx), grad_output.options());
}

std::tuple<Tensor, Tensor> mkldnn_linear_backward_weights(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight, bool bias_defined) {
  auto grad_output_reshaped = grad_output.dim() > 2 ?
    grad_output.reshape({-1, grad_output.size(grad_output.dim() - 1)}) : grad_output;
  auto input_reshaped = input.dim() > 2 ? input.reshape({-1, input.size(input.dim() - 1)}) : input;

  ideep::tensor& grady = itensor_from_mkldnn(grad_output_reshaped);
  ideep::tensor& x = itensor_from_mkldnn(input_reshaped);
  ideep::tensor gradw, gradb;
  if (bias_defined) {
    ideep::inner_product_backward_weights::compute(x, grady, gradw, gradb);
  } else {
    ideep::inner_product_backward_weights::compute(x, grady, gradw);
  }

  if (weight.is_mkldnn()) {
    return std::tuple<Tensor, Tensor>{
      new_with_itensor_mkldnn(std::move(gradw), weight.options()),
      new_with_itensor_mkldnn(std::move(gradb), weight.options())};
  } else {
    return std::tuple<Tensor, Tensor>{
      mkldnn_to_dense(new_with_itensor_mkldnn(std::move(gradw), weight.options()), weight.scalar_type()),
      mkldnn_to_dense(new_with_itensor_mkldnn(std::move(gradb), weight.options()), weight.scalar_type())};
  }
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_linear_backward(
    const Tensor& input, const Tensor& grad_output,
    const Tensor& weight, std::array<bool,3> output_mask) {
  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = at::mkldnn_linear_backward_input(input.sizes(), grad_output, weight);
  }
  if (output_mask[1] || output_mask[2]) {
    std::tie(grad_weight, grad_bias) = at::mkldnn_linear_backward_weights(grad_output, input, weight, output_mask[2]);
  }
  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
