#pragma once

#include <ATen/ATen.h>
#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ideep.hpp>

namespace at { namespace native {

// Construct aten MKL-DNN tensor given an ideep tensor
Tensor new_with_itensor_mkldnn(ideep::tensor&& it, const TensorOptions& options);

// Retrieve `ideep::tensor` from MKL-DNN tensor
ideep::tensor& itensor_from_mkldnn(const Tensor& mkldnn_tensor);

// Construct an `ideep::tensor` "view" from dense tensor, note the
// ideep::tensor will share the underlying buffer
ideep::tensor itensor_view_from_dense(const Tensor& tensor);

// Helper function for getting an ideep tensor out of an aten Tensor.
ideep::tensor itensor_from_tensor(const at::Tensor& tensor);

// Retrieve mkldnn data type from cpu ScalarType
ideep::tensor::data_type get_mkldnn_dtype(ScalarType dtype);

// Calculate MKL-DNN tensor scales from QTensor scales
ideep::scale_t ConvertScales(const std::vector<double> &scales_z);

}}

#endif // AT_MKLDNN_ENABLED
