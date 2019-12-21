#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qmkldnn_utils.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>

namespace at {
namespace native {
namespace fbgemm_utils {

#ifdef USE_FBGEMM
std::tuple<at::Tensor, c10::optional<Tensor>> fbgemm_linear_unpack(
    at::Tensor packed_weight) {
  // Pull out the PackBMatrix instance from the owning tensor.
  auto& pack_ptr =
      cpp_custom_type_hack::cast<PackedLinearWeight>(packed_weight);
  auto packB = pack_ptr.w.get();

  int64_t N = static_cast<int64_t>(packB->numCols());
  int64_t K = static_cast<int64_t>(packB->numRows());

  Tensor weight_origin;
  if (pack_ptr.q_scheme == c10::kPerTensorAffine) {
    weight_origin = at::_empty_affine_quantized(
        {N, K},
        at::device(c10::kCPU).dtype(c10::kQInt8),
        pack_ptr.w_scale[0],
        pack_ptr.w_zp[0]);
  } else if (pack_ptr.q_scheme == c10::kPerChannelAffine) {
    auto scales = from_blob(
        pack_ptr.w_scale.data(),
        pack_ptr.w_scale.size(),
        device(c10::kCPU).dtype(c10::kFloat));
    auto zero_points = from_blob(
        pack_ptr.w_zp.data(), pack_ptr.w_zp.size(), device(kCPU).dtype(kInt));

    weight_origin = at::_empty_per_channel_affine_quantized(
        {N, K},
        scales.toType(c10::kDouble),
        zero_points.toType(c10::kLong),
        0, // The output channel axis is 0
        device(c10::kCPU).dtype(c10::kQInt8));
  }

  int8_t* weight_ptr_int8 =
      reinterpret_cast<int8_t*>(weight_origin.data_ptr<c10::qint8>());

  // packB->printPackedMatrix("packedB inside fbgemm_unpack
  // (QLinearUnpackWeightInt8): ");
  packB->unpack(weight_ptr_int8);

  return std::tuple<at::Tensor, c10::optional<Tensor>>(
      weight_origin, pack_ptr.bias);
}
#endif // USE_FBGEMM
}

namespace {

class QLinearUnpackWeightInt8 final : public c10::OperatorKernel {
 public:
#ifdef USE_PYTORCH_QNNPACK
  std::tuple<at::Tensor, c10::optional<Tensor>> qnnpack_linear_unpack(
      at::Tensor packed_weight) {
    auto& pack_ptr =
        cpp_custom_type_hack::cast<PackedLinearWeightsQnnp>(packed_weight);
    return std::tuple<at::Tensor, c10::optional<Tensor>>(
        pack_ptr.orig_weight, pack_ptr.bias);
  }
#endif // USE_PYTORCH_QNNPACK
  std::tuple<at::Tensor, c10::optional<Tensor>> operator()(
      at::Tensor packed_weight) {
    auto& ctx = at::globalContext();

#ifdef USE_FBGEMM
#if AT_MKLDNN_ENABLED()
    if (cpp_custom_type_hack::is_type<PackedWeightQmkldnn>(
            packed_weight)) {
      return mkldnn_linear_unpack(packed_weight);
    }
#endif
    if (cpp_custom_type_hack::is_type<PackedLinearWeight>(
            packed_weight)) {
      return fbgemm_utils::fbgemm_linear_unpack(packed_weight);
    }
#endif
#ifdef USE_PYTORCH_QNNPACK
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      return qnnpack_linear_unpack(packed_weight);
    }
#endif
    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::linear_unpack ",
        toString(ctx.qEngine()));
  }
};

static auto registry = c10::RegisterOperators().op(
    "quantized::linear_unpack(Tensor W_prepack) -> (Tensor W_origin, Tensor? B_origin)",
    c10::RegisterOperators::options().kernel<QLinearUnpackWeightInt8>(
        TensorTypeId::CPUTensorId));

} // namespace
} // namespace native
} // namespace at
