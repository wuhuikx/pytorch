#include <torch/csrc/distributed/rpc/request_callback_impl.h>
#include <c10/util/C++17.h>
#include <torch/csrc/distributed/autograd/context/dist_autograd_container.h>
#include <torch/csrc/distributed/autograd/context/dist_autograd_context.h>
#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/csrc/distributed/rpc/future_message.h>
#include <torch/csrc/distributed/rpc/python_remote_call.h>
#include <torch/csrc/distributed/rpc/python_rpc_handler.h>
#include <torch/csrc/distributed/rpc/python_udf_call.h>
#include <torch/csrc/distributed/rpc/python_udf_resp.h>
#include <torch/csrc/distributed/rpc/rpc_with_autograd.h>
#include <torch/csrc/distributed/rpc/rref.h>
#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/rref_proto.h>
#include <torch/csrc/distributed/rpc/script_call.h>
#include <torch/csrc/distributed/rpc/script_remote_call.h>
#include <torch/csrc/distributed/rpc/script_resp.h>
#include <torch/csrc/distributed/rpc/utils.h>

namespace torch {
namespace distributed {
namespace rpc {

using namespace torch::distributed::autograd;

std::unique_ptr<RpcCommandBase> RequestCallbackImpl::processRpc(
    RpcCommandBase& rpc,
    MessageType messageType) const {
  // TODO: RpcCommandBase should have an abstract execute() method that we can
  // call here instead of having another switch statement here. Even better we
  // could have abstract classes RpcRequest and RpcResp which inherit from
  // RpcCommandBase and RpcRequest declares the abstract method execute() that
  // we can call here. RpcResponse could have an abstract method to convert it
  // to a python object.
  switch (messageType) {
    case MessageType::SCRIPT_CALL: {
      auto& scriptCall = static_cast<ScriptCall&>(rpc);

      // sc is only alive within this block, use reference to avoid copy
      auto& stack = scriptCall.stackRef();
      scriptCall.op()->getOperation()(stack);

      TORCH_INTERNAL_ASSERT(
          stack.size() == 1,
          "Return value of a builtin operator or a "
          "TorchScript function should be a single IValue, got a vector of "
          "size ",
          stack.size());

      return c10::guts::make_unique<ScriptResp>(std::move(stack.front()));
    }
    case MessageType::PYTHON_CALL: {
      auto& pyCall = static_cast<PythonUDFCall&>(rpc);
      std::vector<torch::Tensor> responseTensorTable;
      auto payload = PythonRpcHandler::getInstance().generatePythonUDFResult(
          pyCall.pickledPayload(), pyCall.tensors(), responseTensorTable);
      return c10::guts::make_unique<PythonUDFResp>(
          std::move(payload), std::move(responseTensorTable));
    }
    case MessageType::SCRIPT_REMOTE_CALL: {
      auto& src = static_cast<ScriptRemoteCall&>(rpc);
      auto& ctx = RRefContext::getInstance();

      auto ownerRRef = ctx->getOrCreateOwnerRRef<IValue>(src.retRRefId());

      // TODO: make this asynchronous
      // src is only alive within this block, use reference to avoid copy
      auto& stack = src.stackRef();
      src.op()->getOperation()(stack);
      TORCH_INTERNAL_ASSERT(
          stack.size() == 1,
          "Return value of a builtin operator or a "
          "TorchScript function should be a single IValue, got a vector of "
          "size ",
          stack.size());

      ownerRRef->setValue(std::move(stack.front()));
      ctx->addForkOfOwner(src.retRRefId(), src.retForkId());
      return c10::guts::make_unique<RemoteRet>(
          src.retRRefId(), src.retForkId());
    }
    case MessageType::PYTHON_REMOTE_CALL: {
      auto& prc = static_cast<PythonRemoteCall&>(rpc);

      auto rrefId = RRefId::fromIValue(prc.retRRefId());
      auto forkId = ForkId::fromIValue(prc.retForkId());
      auto& ctx = RRefContext::getInstance();

      auto ownerRRef = ctx->getOrCreateOwnerRRef<py::object>(rrefId);
      ownerRRef->setValue(
          PythonRpcHandler::getInstance().runPythonUDF(prc.serializedPyObj()));
      ctx->addForkOfOwner(rrefId, forkId);
      return c10::guts::make_unique<RemoteRet>(rrefId, forkId);
    }
    case MessageType::SCRIPT_RREF_FETCH_CALL: {
      auto& srf = static_cast<ScriptRRefFetchCall&>(rpc);
      auto& ctx = RRefContext::getInstance();
      // TODO: make this asynchronous
      std::shared_ptr<OwnerRRef<IValue>> rref =
          ctx->getOrCreateOwnerRRef<IValue>(srf.rrefId());
      return c10::guts::make_unique<RRefFetchRet>(
          RRefFetchRet({rref->getValue()}));
    }
    case MessageType::PYTHON_RREF_FETCH_CALL: {
      auto& prf = static_cast<PythonRRefFetchCall&>(rpc);
      auto& ctx = RRefContext::getInstance();
      // TODO: make this asynchronous
      std::shared_ptr<OwnerRRef<py::object>> rref =
          ctx->getOrCreateOwnerRRef<py::object>(prf.rrefId());
      SerializedPyObj result =
          PythonRpcHandler::getInstance().serialize(rref->getValue());
      return c10::guts::make_unique<RRefFetchRet>(
          RRefFetchRet(result.toIValues()));
    }
    case MessageType::RREF_USER_DELETE: {
      auto& rud = static_cast<RRefUserDelete&>(rpc);
      auto& ctx = RRefContext::getInstance();
      ctx->delForkOfOwner(rud.rrefId(), rud.forkId());
      return c10::guts::make_unique<RRefAck>();
    }
    case MessageType::RREF_CHILD_ACCEPT: {
      auto& rca = static_cast<RRefChildAccept&>(rpc);
      auto& ctx = RRefContext::getInstance();
      ctx->delPendingChild(rca.forkId());
      return c10::guts::make_unique<RRefAck>();
    }
    case MessageType::RREF_FORK_REQUEST: {
      auto& rfr = static_cast<RRefForkRequest&>(rpc);
      auto& ctx = RRefContext::getInstance();
      ctx->addForkOfOwner(rfr.rrefId(), rfr.forkId());
      return c10::guts::make_unique<RRefAck>();
    }
    case MessageType::MESSAGE_WITH_AUTOGRAD_REQ: {
      auto& rpcWithAutograd = static_cast<RpcWithAutograd&>(rpc);
      const auto& autogradMetadata = rpcWithAutograd.autogradMetadata();

      // Attach 'recv' autograd function.
      DistAutogradContext* autogradContext = addRecvRpcBackward(
          rpcWithAutograd.autogradMetadata(), rpcWithAutograd.tensors());

      // Process the original RPC.
      auto wrappedMessageType = rpcWithAutograd.wrappedMessageType();
      auto wrappedRpcResponse =
          processRpc(rpcWithAutograd.wrappedRpc(), wrappedMessageType);

      // Wrap the response with autograd, need a new autograd message id for
      // each send/recv pair.
      auto& autogradContainer = DistAutogradContainer::getInstance();
      AutogradMetadata responseAutogradMetadata(
          autogradMetadata.autogradContextId,
          autogradContainer.newAutogradMessageId());

      auto response = c10::guts::make_unique<RpcWithAutograd>(
          MessageType::MESSAGE_WITH_AUTOGRAD_RESP,
          responseAutogradMetadata,
          std::move(wrappedRpcResponse));

      // Attach the 'send' autograd function if needed.
      if (autogradContext != nullptr) {
        addSendRpcBackward(
            *autogradContext, responseAutogradMetadata, response->tensors());
      }
      return std::move(response);
    }
    default: {
      TORCH_INTERNAL_ASSERT(
          false, "Request type ", messageType, " not supported.");
    }
  }
}

Message RequestCallbackImpl::processMessage(Message& request) const {
  std::unique_ptr<RpcCommandBase> rpc = deserializeRequest(request);
  auto response = processRpc(*rpc, request.type());
  if (response == nullptr) {
    return Message();
  }
  auto responseMessage = std::move(*response).toMessage();
  responseMessage.setId(request.id());
  return responseMessage;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
