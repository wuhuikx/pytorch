#include "caffe2/core/tensor.h"
#include "caffe2/core/workspace.h"

#include "test_utils.h"

namespace {
template <typename T>
void assertTensorEqualsWithType(
    const caffe2::TensorCPU& tensor1,
    const caffe2::TensorCPU& tensor2) {
  CAFFE_ENFORCE_EQ(tensor1.sizes(), tensor2.sizes());
  for (auto idx = 0; idx < tensor1.numel(); ++idx) {
    CAFFE_ENFORCE_EQ(tensor1.data<T>()[idx], tensor2.data<T>()[idx]);
  }
}
} // namespace

namespace caffe2 {
namespace testing {

void assertTensorEquals(const TensorCPU& tensor1, const TensorCPU& tensor2) {
  CAFFE_ENFORCE_EQ(tensor1.sizes(), tensor2.sizes());
  if (tensor1.IsType<float>()) {
    CAFFE_ENFORCE(tensor2.IsType<float>());
    assertTensorEqualsWithType<float>(tensor1, tensor2);
  } else if (tensor1.IsType<int>()) {
    CAFFE_ENFORCE(tensor2.IsType<int>());
    assertTensorEqualsWithType<int>(tensor1, tensor2);
  } else if (tensor1.IsType<int64_t>()) {
    CAFFE_ENFORCE(tensor2.IsType<int64_t>());
    assertTensorEqualsWithType<int64_t>(tensor1, tensor2);
  }
  // Add more types if needed.
}

void assertTensorListEquals(
    const std::vector<std::string>& tensorNames,
    const Workspace& workspace1,
    const Workspace& workspace2) {
  for (const string& tensorName : tensorNames) {
    CAFFE_ENFORCE(workspace1.HasBlob(tensorName));
    CAFFE_ENFORCE(workspace2.HasBlob(tensorName));
    auto& tensor1 = getTensor(workspace1, tensorName);
    auto& tensor2 = getTensor(workspace2, tensorName);
    assertTensorEquals(tensor1, tensor2);
  }
}

const caffe2::Tensor& getTensor(
    const caffe2::Workspace& workspace,
    const std::string& name) {
  return workspace.GetBlob(name)->Get<caffe2::Tensor>();
}

caffe2::Tensor* createTensor(
    const std::string& name,
    caffe2::Workspace* workspace) {
  return BlobGetMutableTensor(workspace->CreateBlob(name), caffe2::CPU);
}

caffe2::OperatorDef* createOperator(
    const std::string& type,
    const std::vector<string>& inputs,
    const std::vector<string>& outputs,
    caffe2::NetDef* net) {
  auto* op = net->add_op();
  op->set_type(type);
  for (const auto& in : inputs) {
    op->add_input(in);
  }
  for (const auto& out : outputs) {
    op->add_output(out);
  }
  return op;
}

NetMutator& NetMutator::newOp(
    const std::string& type,
    const std::vector<string>& inputs,
    const std::vector<string>& outputs) {
  createOperator(type, inputs, outputs, net_);
  return *this;
}

} // namespace testing
} // namespace caffe2
