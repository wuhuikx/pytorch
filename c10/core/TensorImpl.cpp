#include <c10/core/TensorImpl.h>

#include <c10/core/Backend.h>
#include <c10/core/WrapDimMinimal.h>
#include <c10/core/impl/LocalTensorTypeSet.h>
#include <c10/util/Optional.h>

C10_DEFINE_bool(
    caffe2_keep_on_shrink,
    true,
    "If set, keeps memory when a tensor is shrinking its size.");

C10_DEFINE_int64(
    caffe2_max_keep_on_shrink_memory,
    LLONG_MAX,
    "The maximum memory in bytes to keep on shrink, if the difference between "
    "tensor sizes is bigger than this then tensor will be reset.");

namespace c10 {

const char * const TensorImpl::err_msg_tensor_metadata_change_not_allowed =
    "is not allowed on a Tensor created from .data or .detach().\n"
    "If your intent is to change the metadata of a Tensor (such as sizes / strides / storage / storage_offset)\n"
    "without autograd tracking the change, remove the .data / .detach() call and wrap the change in a `with torch.no_grad():` block.\n"
    "For example, change:\n"
    "    x.data.set_(y)\n"
    "to:\n"
    "    with torch.no_grad():\n"
    "        x.set_(y)";

at::Tensor& TensorImpl::grad() {
  if (autograd_meta()) {
    return autograd_meta()->grad();
  } else {
    AT_ERROR("grad is not implemented for Tensor");
  }
}

const at::Tensor& TensorImpl::grad() const {
  if (autograd_meta()) {
    return autograd_meta()->grad();
  } else {
    AT_ERROR("grad is not implemented for Tensor");
  }
}

TensorImpl::TensorImpl(Storage&& storage, TensorTypeSet type_set)
    : TensorImpl(std::move(storage), type_set, storage.dtype(), storage.device()) {}

TensorImpl::TensorImpl(TensorTypeSet type_set, const caffe2::TypeMeta& data_type, c10::optional<c10::Device> device_opt)
    : TensorImpl({}, type_set, data_type, std::move(device_opt)) {}

TensorImpl::TensorImpl(Storage&& storage, TensorTypeSet type_set, const caffe2::TypeMeta& data_type,
                       c10::optional<c10::Device> device_opt)
    : storage_(std::move(storage)),
      sizes_{0},
      storage_offset_(0),
      numel_(0),
      data_type_(data_type),
      device_opt_(device_opt),
      type_set_(type_set.remove(TensorTypeId::VariableTensorId)) {
  if (!type_set.empty()) {
    AT_ASSERT(data_type.id() ==  caffe2::TypeIdentifier::uninitialized() ||
              device_opt_.has_value());
    // UndefinedTensorImpl is a singleton, so we skip logging it
    C10_LOG_API_USAGE_ONCE("tensor.create");
  }
  // we would also like to check that non-cpu devices have an index, but some Caffe2 operators create
  // Storages with default devices.
  strides_.push_back(1);
}

IntArrayRef TensorImpl::sizes() const {
  return sizes_;
}

IntArrayRef TensorImpl::strides() const {
  return strides_;
}

bool TensorImpl::compute_contiguous() const {
  bool is_contiguous = true;
  if (is_empty())
    return is_contiguous;
  int64_t z = 1;
  for (int64_t d = dim() - 1; d >= 0; d--) {
    if (size(d) != 1) {
      if (stride(d) == z) {
        z *= size(d);
      } else {
        is_contiguous = false;
        break;
      }
    }
  }
  return is_contiguous;
}

bool TensorImpl::compute_channels_last_contiguous() const {
  if (dim() == 4) {
    int64_t expected = 1;
    for (auto& d : {1, 3, 2, 0}) {
      if (size(d) != 1) {
        if (stride(d) == expected) {
          expected *= size(d);
        } else {
          return false;
        }
      }
    }
    return true;
  }
  return false;
}

bool TensorImpl::compute_strides_like_channels_last() const {
  if (dim() == 4) {
    int64_t min = 0;
    for (auto& d : {1, 3, 2, 0}) {
      if (size(d) != 1) {
        if (stride(d) > min) {
          min = stride(d);
        } else {
          return false;
        }
      }
    }
    return true;
  }
  return false;
}

bool TensorImpl::compute_non_overlapping_and_dense() const {
  if (dim() == 1) {
    return size(0) < 2 || stride(0) == 1;
  }
  SmallVector<int64_t,5> perm;
  perm.resize(dim());
  for (int64_t i = 0; i < dim(); i ++) {
    perm[i] = i;
  }
  // Sort by strides, leaving 0 and 1 sized dims at the end of the array
  std::sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
      if (sizes_[a] < 2) {
        return false;
      } else if (sizes_[b] < 2) {
        return true;
      }
      return strides_[a] < strides_[b];
  });
  auto require_stride = 1;
  for (int64_t i = 0; i < dim(); i ++) {
    if (sizes_[perm[i]] < 2) {
      return true;
    }
    if (strides_[perm[i]] != require_stride) {
      return false;
    }
    require_stride *= sizes_[perm[i]];
  }
  return true;
}

void TensorImpl::release_resources() {
  autograd_meta_.reset();
  if (storage_) {
    storage_ = {};
  }
}

int64_t TensorImpl::dim() const {
  return sizes_.size();
}

int64_t TensorImpl::size(int64_t d) const {
  d = at::maybe_wrap_dim(d, dim(), false);
  return sizes_[d];
}

int64_t TensorImpl::stride(int64_t d) const {
  d = at::maybe_wrap_dim(d, dim(), false);
  return strides_[d];
}

TensorImpl* TensorImpl::maybe_zero_dim(bool condition_when_zero_dim) {
  bool set_zero_dim = condition_when_zero_dim && this->sizes().size() == 1 && this->size(0) == 1;
  if (set_zero_dim) {
    resize_dim(0);
  }
  return this;
}

bool TensorImpl::has_storage() const {
  return storage_;
}

bool TensorImpl::is_contiguous(at::MemoryFormat memory_format) const {
#ifdef DEBUG
  AT_ASSERT(compute_contiguous() == is_contiguous_);
#endif
  if (memory_format == at::MemoryFormat::ChannelsLast) {
      return is_channels_last_contiguous_;
  }
  return is_contiguous_;
}

const Storage& TensorImpl::storage() const {
  return storage_;
}

static void deletePlacementDeleteContext(void* ptr) {
  delete static_cast<PlacementDeleteContext*>(ptr);
}

at::DataPtr PlacementDeleteContext::makeDataPtr(
    at::DataPtr&& data_ptr,
    PlacementDtor placement_dtor,
    size_t size,
    at::Device device) {
  auto* ptr = data_ptr.get();
  return {ptr,
          new PlacementDeleteContext(std::move(data_ptr), placement_dtor, size),
          &deletePlacementDeleteContext,
          device};
}

AutogradMetaInterface::~AutogradMetaInterface() {}

void TensorImpl::set_requires_grad(bool requires_grad) {
  TORCH_INTERNAL_ASSERT(autograd_meta(), "set_requires_grad is not implemented for Tensor");
  autograd_meta()->set_requires_grad(requires_grad, this);
}

bool TensorImpl::requires_grad() const {
  TORCH_INTERNAL_ASSERT(autograd_meta(), "requires_grad is not implemented for Tensor");
  return autograd_meta()->requires_grad();
}

void TensorImpl::set_autograd_meta(std::unique_ptr<c10::AutogradMetaInterface> autograd_meta) {
  autograd_meta_ = std::move(autograd_meta);
  if (autograd_meta_) {
    type_set_ = type_set_.add(TensorTypeId::VariableTensorId);
  } else {
    type_set_ = type_set_.remove(TensorTypeId::VariableTensorId);
  }
}

c10::AutogradMetaInterface* TensorImpl::autograd_meta() const {
  return autograd_meta_.get();
}

std::unique_ptr<c10::AutogradMetaInterface> TensorImpl::detach_autograd_meta() {
  type_set_ = type_set_.remove(TensorTypeId::VariableTensorId);
  return std::move(autograd_meta_);
}

void TensorImpl::copy_tensor_metadata(
    const TensorImpl* src_impl,
    TensorImpl* dest_impl,
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) {
  dest_impl->storage_ = src_impl->storage_;
  dest_impl->sizes_ = src_impl->sizes_;
  dest_impl->strides_ = src_impl->strides_;
  dest_impl->storage_offset_ = src_impl->storage_offset_;
  dest_impl->data_type_ = src_impl->data_type_;
  dest_impl->device_opt_ = src_impl->device_opt_;
  // This may temporarily violate invariant that
  // type_set_.has(VariableTensorId) iff autograd_meta_ != nullptr...
  dest_impl->type_set_ = src_impl->type_set_;
  // ...so refresh Variable in autograd_meta_
  if (dest_impl->autograd_meta_) {
    dest_impl->type_set_ = dest_impl->type_set_.add(TensorTypeId::VariableTensorId);
  } else {
    dest_impl->type_set_ = dest_impl->type_set_.remove(TensorTypeId::VariableTensorId);
  }
  dest_impl->is_contiguous_ = src_impl->is_contiguous_;
  dest_impl->is_channels_last_contiguous_ = src_impl->is_channels_last_contiguous_;
  dest_impl->is_channels_last_ = src_impl->is_channels_last_;
  dest_impl->is_non_overlapping_and_dense_ = src_impl->is_non_overlapping_and_dense_;
  dest_impl->is_wrapped_number_ = src_impl->is_wrapped_number_;
  dest_impl->reserved_ = src_impl->reserved_;
  dest_impl->set_version_counter(version_counter);
  dest_impl->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
  if (src_impl->named_tensor_meta_ != nullptr) {
    dest_impl->named_tensor_meta_ = src_impl->named_tensor_meta_->clone();
  }
}

namespace impl {

namespace {
AutogradMetaFactory* meta_factory = nullptr;
}

void SetAutogradMetaFactory(AutogradMetaFactory* factory) {
  meta_factory = factory;
}
AutogradMetaFactory* GetAutogradMetaFactory() {
  TORCH_CHECK(meta_factory, "Support for autograd has not been loaded; have you linked against libtorch.so?")
  return meta_factory;
}

} // namespace impl

} // namespace c10
