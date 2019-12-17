#pragma once

#include <c10/util/ArrayRef.h>
#include <vector>

namespace at { namespace native {

std::vector<int64_t> conv_output_size(
    IntArrayRef input_size,
    IntArrayRef kernel_size,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation);

std::vector<int64_t> conv_input_size(
    IntArrayRef output_size,
    IntArrayRef kernel_size,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups);

std::vector<int64_t> pool_output_sizes(
    IntArrayRef input_size,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding_l,
    IntArrayRef padding_r,
    IntArrayRef dilation,
    bool ceil_mode);
}}
