#include <torch/csrc/distributed/rpc/rref_impl.h>

#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_autograd.h>
#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/csrc/distributed/rpc/python_rpc_handler.h>
#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/rref_proto.h>
#include <torch/csrc/distributed/rpc/utils.h>
#include <torch/csrc/jit/pybind_utils.h>

namespace torch {
namespace distributed {
namespace rpc {

namespace {

constexpr int OWNER_IDX = 0; // index of ownerId in the tuple
constexpr int RREFID_ON_IDX = 1; // index of RRefId.createdOn_ in the tuple
constexpr int RREFID_ID_IDX = 2; // index of RRefId.localId_ in the tuple
constexpr int FORKID_ON_IDX = 3; // index of ForkId.createdOn_ in the tuple
constexpr int FORKID_ID_IDX = 4; // index of ForkId.localId_ in the tuple
constexpr int PARENT_IDX = 5; // index of parent in the tuple
constexpr int TYPE_IDX = 6; // index of parent in the tuple

// NB: if more fields are added, make sure this field is also bumped
constexpr int RFD_TUPLE_SIZE = 7; // number of RRefForkData fields in py::tuple
} // namespace

std::atomic<local_id_t> RRefContext::nextLocalId_{0};

//////////////////////////  RRefForkData  /////////////////////////////////

RRefForkData::RRefForkData(
    worker_id_t ownerId,
    const RRefId& rrefId,
    const ForkId& forkId,
    worker_id_t parent,
    std::string type_str)
    : ownerId_(ownerId),
      rrefId_(rrefId),
      forkId_(forkId),
      parent_(parent),
      type_str_(std::move(type_str)) {}

py::tuple RRefForkData::toPyTuple() const {
  return py::make_tuple(
      ownerId_,
      rrefId_.createdOn_,
      rrefId_.localId_,
      forkId_.createdOn_,
      forkId_.localId_,
      parent_,
      type_str_);
}

RRefForkData RRefForkData::fromPyTuple(const py::tuple& t) {
  TORCH_INTERNAL_ASSERT(
      t.size() == RFD_TUPLE_SIZE,
      "Pickled RRefForkData must contain 6 numbers.");
  worker_id_t ownerId = t[OWNER_IDX].cast<worker_id_t>();
  // const reference will extend the lifetime of the temporary variable
  const RRefId& rrefId = RRefId(
      t[RREFID_ON_IDX].cast<worker_id_t>(),
      t[RREFID_ID_IDX].cast<local_id_t>());
  const RRefId& forkId = RRefId(
      t[FORKID_ON_IDX].cast<worker_id_t>(),
      t[FORKID_ID_IDX].cast<local_id_t>());

  worker_id_t parent = t[PARENT_IDX].cast<worker_id_t>();
  const std::string& typeStr = t[TYPE_IDX].cast<std::string>();

  return RRefForkData(ownerId, rrefId, forkId, parent, typeStr);
}

//////////////////////////////  RRef  /////////////////////////////////////

RRef::RRef(worker_id_t ownerId, const RRefId& rrefId, TypePtr type)
    : RRefInterface(),
      ownerId_(ownerId),
      rrefId_(rrefId),
      type_(std::move(type)) {}

RRefForkData RRef::fork() const {
  auto& ctx = RRefContext::getInstance();
  return RRefForkData(
      ownerId_,
      rrefId_,
      ctx.genGloballyUniqueId(),
      ctx.getWorkerId(),
      type_->str());
}

//////////////////////////  UserRRef  /////////////////////////////////////

UserRRef::UserRRef(
    worker_id_t ownerId,
    const RRefId& rrefId,
    const ForkId& forkId,
    TypePtr type)
    : RRef(ownerId, rrefId, std::move(type)), forkId_(forkId) {
  // Do nothing,
  // (1) If this UserRRef is a fork of an existing RRef, RRefContext will send
  //     a RREF_FORK_REQUEST message to the owner.
  // (2) If this the creator UserRRef, ScriptRemoteCall or PythonRemoteCall will
  //     properly notify the owner.
}

UserRRef::~UserRRef() {
  try {
    RRefContext::getInstance().delUser(ownerId_, rrefId_, forkId_);
  } catch (const std::exception& ex) {
    LOG(ERROR) << "Error occurred when deleting UserRRef instance, "
               << "RRefId = " << rrefId_ << ", ForkId = " << forkId_ << " : "
               << ex.what();
  } catch (...) {
    LOG(ERROR) << "Error occurred when deleting UserRRef instance, "
               << "RRefId = " << rrefId_ << ", ForkId = " << forkId_ << " : "
               << "unknown error";
  }
}

const ForkId& UserRRef::forkId() const {
  return forkId_;
}

IValue UserRRef::toHere() {
  auto agent = RpcAgent::getCurrentRpcAgent();

  // ScriptRRefFetchCall message always carries autograd context id even if
  // the message itself does not contain any tensor, because the response would
  // potentially contain tensors.
  Message msgToSend;

  if (isPyObj()) {
    msgToSend = PythonRRefFetchCall(ownerId_, rrefId()).toMessage();
  } else {
    msgToSend = ScriptRRefFetchCall(ownerId_, rrefId()).toMessage();
  }

  auto futureResponse = autograd::sendMessageWithAutograd(
      *agent,
      agent->getWorkerInfo(ownerId_),
      std::move(msgToSend),
      true /* forceGradRecording */);

  const Message& message = futureResponse->wait();
  MessageType msgType = message.type();
  auto response = deserializeResponse(message, msgType);
  TORCH_INTERNAL_ASSERT(
      msgType == MessageType::SCRIPT_RREF_FETCH_RET ||
          msgType == MessageType::PYTHON_RREF_FETCH_RET,
      "Message type should either be SCRIPT_RREF_FETCH_RET "
      "or PYTHON_RREF_FETCH_RET");
  RpcCommandBase& rpc = *response;
  if (isPyObj()) {
    auto& rfr = static_cast<PythonRRefFetchRet&>(rpc);
    return jit::toIValue(
        PythonRpcHandler::getInstance().deserialize(
            SerializedPyObj::fromIValues(rfr.values())),
        PyObjectType::get());
  } else {
    auto& rfr = static_cast<ScriptRRefFetchRet&>(rpc);
    return rfr.values().front();
  }
}

//////////////////////////  OwnerRRef  /////////////////////////////////////

const IValue& OwnerRRef::getValue() const {
  std::unique_lock<std::mutex> lock(mutex_);
  valueCV_.wait(lock, [this] { return value_.has_value(); });
  return value_.value();
}

bool OwnerRRef::hasValue() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return value_.has_value();
}

std::shared_ptr<FutureMessage> OwnerRRef::getFuture() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (future_.get()) {
    return future_;
  }
  future_ = std::make_shared<FutureMessage>();
  std::shared_ptr<FutureMessage> ret = future_;
  if (value_.has_value()) {
    lock.unlock();
    ret->markCompleted(Message());
  }
  return ret;
}

void OwnerRRef::setValue(IValue&& value) {
  std::unique_lock<std::mutex> lock(mutex_);
  value_ = std::move(value);
  std::shared_ptr<FutureMessage> future;
  future.swap(future_);
  lock.unlock();
  valueCV_.notify_all();
  if (future.get() && !future->completed()) {
    future->markCompleted(Message());
  }
}

} // namespace rpc
} // namespace distributed
} // namespace torch
