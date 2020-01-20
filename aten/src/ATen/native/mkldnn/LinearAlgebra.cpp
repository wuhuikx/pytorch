#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

Tensor bmm_mkldnn(
    const Tensor& self,
    const Tensor& mat2) {
  TORCH_CHECK(false, "bmm_mkldnn: ATen not compiled with MKLDNN support");
}

Tensor& bmm_out_mkldnn(
    Tensor &result, 
    const Tensor& batch1, 
    const Tensor& batch2) {
  TORCH_CHECK(false, "bmm_out_mkldnn: ATen not compiled with MKLDNN support");
}

Tensor mm_mkldnn(
    const Tensor& self,
    const Tensor& mat2) {
  TORCH_CHECK(false, "mm_mkldnn: ATen not compiled with MKLDNN support");
}

Tensor& mm_out_mkldnn(
    Tensor& result,
    const Tensor& self,
    const Tensor& mat2) {
  TORCH_CHECK(false, "mm_out_mkldnn: ATen not compiled with MKLDNN support");
}

Tensor baddbmm_mkldnn(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor & batch2,
    Scalar beta,
    Scalar alpha) {
  TORCH_CHECK(false, "baddbmm_mkldnn: ATen not compiled with MKLDNN support");
}

Tensor& baddbmm_out_mkldnn(
    Tensor &result, 
    const Tensor& self, 
    const Tensor& batch1, 
    const Tensor& batch2, 
    Scalar beta, 
    Scalar alpha) {
  TORCH_CHECK(false, "baddbmm_out_mkldnn: ATen not compiled with MKLDNN support");
}

Tensor baddbmm__mkldnn(
    Tensor& self,
    const Tensor& batch1,
    const Tensor & batch2,
    Scalar beta,
    Scalar alpha) {
  TORCH_CHECK(false, "baddbmm__mkldnn: ATen not compiled with MKLDNN support");
}

Tensor addmm_mkldnn(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor & batch2,
    Scalar beta,
    Scalar alpha) {
  TORCH_CHECK(false, "addmm_mkldnn: ATen not compiled with MKLDNN support");
}

Tensor& addmm_out_mkldnn(
    Tensor& result,
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    Scalar beta,
    Scalar alpha) {      
  TORCH_CHECK(false, "addmm_out_mkldnn: ATen not compiled with MKLDNN support");
}

Tensor addmm__mkldnn(
    Tensor& self,
    const Tensor& batch1,
    const Tensor & batch2,
    Scalar beta,
    Scalar alpha) {
  TORCH_CHECK(false, "addmm__mkldnn: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at {
namespace native {


Tensor bmm_mkldnn(
    const Tensor& self, 
    const Tensor& mat2) {
  //TORCH_CHECK(self.is_mkldnn(),
  //    "bmm_mkldnn: input needs to be mkldnn layout");
  //TORCH_CHECK(self.dim() == 3 && mat2.dim() == 3,
  //    "bmm_mkldnn: only support inputs with 3 dim");
  //TORCH_CHECK(self.size(0) == mat2.size(0) && self.size(2) == mat2.size(1),
  //    "bmm_mkldnn: inputs dims not match");
  //IntArrayRef size_ = self.sizes();
  //size_[-1] = mat2.sizes()[-1];
  //Tensor result = empty_mkldnn(size_, self.options());
  const ideep::tensor x = itensor_from_mkldnn(self);
  const ideep::tensor w = itensor_from_mkldnn(mat2);
  ideep::tensor y;
  ideep::matmul_forward::compute(x, w, y);
  return new_with_itensor_mkldnn(std::move(y), self.options());
  //return at::native::bmm_out_mkldnn(result, self, mat2);
}

Tensor& bmm_out_mkldnn(
    Tensor &result, 
    const Tensor& batch1, 
    const Tensor& batch2) {
  const ideep::tensor x = itensor_from_mkldnn(batch1);
  const ideep::tensor w = itensor_from_mkldnn(batch2);
  ideep::tensor& y = itensor_from_mkldnn(result);
  ideep::matmul_forward::compute(x, w, y);
  return result;
}

Tensor mm_mkldnn(
    const Tensor& self,
    const Tensor& mat2) {
  return at::native::bmm_mkldnn(self, mat2);
}

Tensor& mm_out_mkldnn(
    Tensor& result,
    const Tensor& self,
    const Tensor& mat2) {
  return at::native::bmm_out_mkldnn(result, self, mat2);
}
/*
Tensor baddmm_mkldnn(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor & batch2,
    Scalar beta,
    Scalar alpha) {
    const ideep::tensor x = itensor_from_mkldnn(batch1);
    const ideep::tensor w = itensor_from_mkldnn(batch2);
    ideep::tensor y = itensor_from_mkldnn(self);
    float dst_coeff = alpha.to<float>();
    float bias_coeff = 1.0f;
    float sum_coeff = beta.to<float>();
    auto attr_ = ideep::attr_t::fuse_sum();
    ideep::matmul_forward::compute(x, w, y, dst_coeff, bias_coeff, sum_coeff,
        ideep::scale_t(), ideep::scale_t(), ideep::scale_t(), attr_);
    return self;
}

Tensor baddmm_mkldnn(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor & batch2,
    Scalar beta,
    Scalar alpha) {
    Tensor result = empty_mkldnn(self.sizes(), self.options());
    return at::native::baddbmm_out_mkldnn(result, self, batch1, batch2, beta, alpha);
}

Tensor& baddbmm_out_mkldnn(
    Tensor &result, 
    const Tensor& self, 
    const Tensor& batch1, 
    const Tensor& batch2, 
    Scalar beta, 
    Scalar alpha) {
    const ideep::tensor x = itensor_from_mkldnn(batch1);
    const ideep::tensor w = itensor_from_mkldnn(batch2);
    const ideep::tensor bias = itensor_from_mkldnn(self);
    ideep::tensor& y = itensor_from_mkldnn(result);

    if (beta == Scalar(1) && alpha == Scalar(1)) {
      ideep::matmul_forward::compute(x, w, bias, y);
      return result;
    }

    ideep::direct_copy::compute(bias, y);
    float dst_coeff = alpha.to<float>();
    float bias_coeff = 1.0f;
    float sum_coeff = beta.to<float>();
    auto attr_ = ideep::attr_t::fuse_sum();
    ideep::matmul_forward::compute(x, w, y, dst_coeff, bias_coeff, sum_coeff,
        ideep::scale_t(), ideep::scale_t(), ideep::scale_t(), attr_);
    return result;
    
}
*/

Tensor baddbmm_mkldnn(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor & batch2,
    Scalar beta,
    Scalar alpha) {
    Tensor result = empty_mkldnn(self.sizes(), self.options());
    return at::native::baddbmm_out_mkldnn(result, self, batch1, batch2, beta, alpha);
}

Tensor& baddbmm_out_mkldnn(
    Tensor &result, 
    const Tensor& self, 
    const Tensor& batch1, 
    const Tensor& batch2, 
    Scalar beta, 
    Scalar alpha) {
    const ideep::tensor x = itensor_from_mkldnn(batch1);
    const ideep::tensor w = itensor_from_mkldnn(batch2);
    const ideep::tensor bias = itensor_from_mkldnn(self);
    ideep::tensor& y = itensor_from_mkldnn(result);

    float dst_coeff = alpha.to<float>();
    float bias_coeff = 1.0f;
    float sum_coeff = beta.to<float>();
    if (dst_coeff == 1.0f  && sum_coeff == 1.0f) {
      ideep::matmul_forward::compute(x, w, bias, y);
      return result;
    }

    ideep::direct_copy::compute(bias, y);
    auto attr_ = ideep::attr_t::fuse_sum();
    ideep::matmul_forward::compute(x, w, y, dst_coeff, bias_coeff, sum_coeff,
        ideep::scale_t(), ideep::scale_t(), ideep::scale_t(), attr_);
    return result;
}

Tensor baddbmm__mkldnn(
    Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    Scalar beta,
    Scalar alpha) {
    const ideep::tensor x = itensor_from_mkldnn(batch1);
    const ideep::tensor w = itensor_from_mkldnn(batch2);
    ideep::tensor y = itensor_from_mkldnn(self);
    float dst_coeff = alpha.to<float>();
    float bias_coeff = 1.0f;
    float sum_coeff = beta.to<float>();
    auto attr_ = ideep::attr_t::fuse_sum();
    ideep::matmul_forward::compute(x, w, y, dst_coeff, bias_coeff, sum_coeff,
        ideep::scale_t(), ideep::scale_t(), ideep::scale_t(), attr_);
    return self;
}

Tensor addmm_mkldnn(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor & batch2,
    Scalar beta,
    Scalar alpha) {
    return at::native::baddbmm_mkldnn(self, batch1, batch2, beta, alpha);
}

Tensor& addmm_out_mkldnn(
    Tensor& result,
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    Scalar beta,
    Scalar alpha) {      
  return at::native::baddbmm_out_mkldnn(result, self, mat1, mat2, beta, alpha);
}

Tensor addmm__mkldnn(
    Tensor& self,
    const Tensor& batch1,
    const Tensor & batch2,
    Scalar beta,
    Scalar alpha) {
  return at::native::baddbmm__mkldnn(self, batch1, batch2, beta, alpha);
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
