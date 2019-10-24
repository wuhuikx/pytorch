#pragma once

#include <c10/util/variant.h>
#include <torch/arg.h>
#include <torch/enum.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/expanding_array.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for a pad functional.
struct TORCH_API PadOptions {
  PadOptions(std::vector<int64_t> pad);

  /// m-elements tuple, where m/2 <= input dimensions and m is even.
  TORCH_ARG(std::vector<int64_t>, pad);

  /// "constant", "reflect", "replicate" or "circular". Default: "constant"
  typedef c10::variant<
    enumtype::kConstant,
    enumtype::kReflect,
    enumtype::kReplicate,
    enumtype::kCircular> mode_t;
  TORCH_ARG(mode_t, mode) = torch::kConstant;

  /// fill value for "constant" padding. Default: 0
  TORCH_ARG(double, value) = 0;
};

// ============================================================================

/// Options for a `D`-dimensional ReflectionPad module.
template <size_t D>
struct TORCH_API ReflectionPadOptions {
  ReflectionPadOptions(ExpandingArray<D*2> padding) : padding_(padding) {}

  /// The size of the padding.
  /// If it is `int`, uses the same padding in all boundaries.
  /// If it is a 2-`tuple` (for ReflectionPad1d), uses (padding_left, padding_right).
  /// If it is a 4-`tuple` (for ReflectionPad2d), uses (padding_left, padding_right, padding_top, padding_bottom).
  TORCH_ARG(ExpandingArray<D*2>, padding);
};

/// `ReflectionPadOptions` specialized for 1-D ReflectionPad.
using ReflectionPad1dOptions = ReflectionPadOptions<1>;

/// `ReflectionPadOptions` specialized for 2-D ReflectionPad.
using ReflectionPad2dOptions = ReflectionPadOptions<2>;

// ============================================================================

/// Options for a `D`-dimensional ReplicationPad module.
template <size_t D>
struct TORCH_API ReplicationPadOptions {
  ReplicationPadOptions(ExpandingArray<D*2> padding) : padding_(padding) {}

  /// The size of the padding.
  /// - If it is `int`, uses the same padding in all boundaries.
  /// - If it is a 2-`tuple` (for ReplicationPad1d), uses (padding_left, padding_right).
  /// - If it is a 4-`tuple` (for ReplicationPad2d), uses (padding_left, padding_right, padding_top, padding_bottom).
  /// - If it is a 6-`tuple` (for ReplicationPad3d), uses
  ///   (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back).
  TORCH_ARG(ExpandingArray<D*2>, padding);
};

/// `ReplicationPadOptions` specialized for 1-D ReplicationPad.
using ReplicationPad1dOptions = ReplicationPadOptions<1>;

/// `ReplicationPadOptions` specialized for 2-D ReplicationPad.
using ReplicationPad2dOptions = ReplicationPadOptions<2>;

/// `ReplicationPadOptions` specialized for 3-D ReplicationPad.
using ReplicationPad3dOptions = ReplicationPadOptions<3>;

} // namespace nn
} // namespace torch
