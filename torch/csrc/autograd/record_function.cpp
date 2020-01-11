#include <torch/csrc/autograd/record_function.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/utils/memory.h>
#include <cstdlib>
#include <random>

namespace torch {
namespace autograd {
namespace profiler {

namespace {

class CallbackManager {
 public:
  void setSamplingProbability(double prob) {
    if (prob == 1.0) {
      sampling_prop_set = false;
    } else {
      TORCH_CHECK(prob >= 0.0 && prob < 1.0);
      sampling_prop_set = true;
    }
    sampling_prob = prob;
  }

  double getSamplingProbability() {
    return sampling_prob;
  }

  bool shouldRunSampledCallbacks() {
    return (num_sampled_callbacks > 0) &&
        (!sampling_prop_set || (sample_zero_one() < sampling_prob));
  }

  void pushCallback(
      RecordFunctionCallback start,
      RecordFunctionCallback end,
      bool needs_inputs,
      bool sampled) {
    start_callbacks.push_back(std::move(start));
    end_callbacks.push_back(std::move(end));
    if (callback_needs_inputs > 0 || needs_inputs) {
      ++callback_needs_inputs;
    }
    is_callback_sampled.push_back(sampled);
    if (sampled) {
      ++num_sampled_callbacks;
    }
  }

  void popCallback() {
    if (start_callbacks.empty()) {
      throw std::runtime_error("Empty callbacks stack");
    }
    start_callbacks.pop_back();
    end_callbacks.pop_back();
    if (callback_needs_inputs > 0) {
      --callback_needs_inputs;
    }
    if (is_callback_sampled.back()) {
      --num_sampled_callbacks;
    }
    is_callback_sampled.pop_back();
  }

  bool hasCallbacks() {
    return !start_callbacks.empty();
  }

  bool needsInputs() {
    return callback_needs_inputs > 0;
  }

  bool hasNonSampledCallbacks() {
    return num_sampled_callbacks < start_callbacks.size();
  }

  std::vector<RecordFunctionCallback> start_callbacks;
  std::vector<RecordFunctionCallback> end_callbacks;
  std::vector<bool> is_callback_sampled;
  size_t num_sampled_callbacks = 0;
  size_t callback_needs_inputs = 0;
  bool sampling_prop_set = false;
  double sampling_prob = 1.0;

  static double sample_zero_one() {
    static thread_local auto gen =
        torch::make_unique<std::mt19937>(std::random_device()());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(*gen);
  }
};

// thread_local_func_ points to the currently active RecordFunction.
thread_local RecordFunction* thread_local_func_ = nullptr;

CallbackManager& manager() {
  static CallbackManager instance;
  return instance;
}

} // namespace

void setSamplingProbability(double prob) {
  manager().setSamplingProbability(prob);
}

double getSamplingProbability() {
  return manager().getSamplingProbability();
}

bool shouldRunSampledCallbacks() {
  return manager().shouldRunSampledCallbacks();
}

void pushCallback(
    RecordFunctionCallback start,
    RecordFunctionCallback end,
    bool needs_inputs,
    bool sampled) {
  manager().pushCallback(
      std::move(start), std::move(end), needs_inputs, sampled);
}

void popCallback() {
  manager().popCallback();
}

bool hasCallbacks() {
  return manager().hasCallbacks();
}

bool needsInputs() {
  return manager().needsInputs();
}

bool hasNonSampledCallbacks() {
  return manager().hasNonSampledCallbacks();
}

void runBeforeCallbacks(RecordFunction* rf, const std::string& funcName) {
  TORCH_INTERNAL_ASSERT(
      rf != nullptr,
      "The RecordFunction passed to before callbacks should not be null.");
  if (hasCallbacks()) {
    auto run_samples = shouldRunSampledCallbacks();
    if (run_samples || hasNonSampledCallbacks()) {
      rf->setRunSampled(run_samples);
      rf->before(funcName);
    }
  }
}

void RecordFunction::before(const char* name, int64_t sequence_nr) {
  if (!hasCallbacks()) {
    return;
  }
  AT_ASSERT(!initialized_);
  name_ = StringView(name);
  sequence_nr_ = sequence_nr;

  initialized_ = true;
  processCallbacks();
}

void RecordFunction::before(std::string name, int64_t sequence_nr) {
  if (!hasCallbacks()) {
    return;
  }
  AT_ASSERT(!initialized_);
  name_ = StringView(std::move(name));
  sequence_nr_ = sequence_nr;

  initialized_ = true;
  processCallbacks();
}

void RecordFunction::before(Node* fn, int64_t sequence_nr) {
  if (!hasCallbacks()) {
    return;
  }
  AT_ASSERT(!initialized_);
  fn_ = fn;
  name_ = StringView(fn->name());
  sequence_nr_ = (sequence_nr >= 0) ? sequence_nr : fn->sequence_nr();

  initialized_ = true;
  processCallbacks();
}

void RecordFunction::processCallbacks() {
  parent_ = thread_local_func_;
  thread_local_func_ = this;

  for (size_t idx = 0; idx < manager().start_callbacks.size(); ++idx) {
    if (!manager().is_callback_sampled[idx] || run_sampled_) {
      manager().start_callbacks[idx](*this);
    }
  }
}

void RecordFunction::setThreadId() {
  auto threadId = torch::autograd::profiler::getThreadId();
  TORCH_INTERNAL_ASSERT(
      threadId != 0,
      "Can only call RecordFunction::setThreadId after RecordFunction::before has been run in this thread.");
  threadId_ = threadId;
}

RecordFunction::~RecordFunction() {
  try {
    end();
  } catch (const std::exception &e) {
    LOG(INFO) << "Exception in RecordFunction::end(): " << e.what();
  }
}

void RecordFunction::end() {
  if (initialized_) {
    for (size_t idx = 0; idx < manager().end_callbacks.size(); ++idx) {
      if (!manager().is_callback_sampled[idx] || run_sampled_) {
        manager().end_callbacks[idx](*this);
      }
    }

    // In the case that RecordFunction::end is called from a different thread,
    // thread_local_func will not be this, so assert that we have overridden the
    // thread id (by ensuring it is nonzero) and thread_local_func is null.
    TORCH_INTERNAL_ASSERT(
        (thread_local_func_ == this) ||
            (thread_local_func_ == nullptr && threadId_ != 0),
        name_,
        ": must be top of stack. If you are calling RecordFunction::end in a"
        "separate thread, call RecordFunction::setThreadId() in the creating"
        "thread.");
    thread_local_func_ = parent_;
    initialized_ = false;
  }
}

RecordFunction* RecordFunction::current() {
  return thread_local_func_;
}

} // namespace profiler
} // namespace autograd
} // namespace torch
