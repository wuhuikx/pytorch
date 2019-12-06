#include "module.h"
#include <torch/csrc/jit/script/jit_exception.h>
#include <torch/csrc/jit/mobile/interpreter.h>
#if defined(PYTORCH_MOBILE_OBSERVER)
#include <torch/csrc/jit/mobile/observer.h>
#endif

namespace torch {
namespace jit {
std::ostream& operator<<(std::ostream& out, Instruction inst);
namespace mobile {

const c10::QualifiedName& Function::qualname() const {
  return name_;
}

const std::string& Function::name() const {
  return name_.name();
}

void CompilationUnit::register_function(std::unique_ptr<Function> fn) {
  methods_.emplace_back(std::move(fn));
}

c10::IValue Module::run_method(const std::string& method_name, Stack stack) {
#if defined(PYTORCH_MOBILE_OBSERVER)
  auto debug_info = std::make_shared<MobileDebugInfo>();
  debug_info->setModelName(name());
  debug_info->setMethodName(method_name);
  at::setThreadLocalDebugInfo(debug_info);

  auto observer = torch::observerConfig().getModuleObserver();
  if (observer) {
    observer->onEnter();
  }
#endif

  auto m = find_method(method_name);
  stack.insert(stack.begin(), object_);
  m->run(stack);
  c10::IValue result = stack.front();

#if defined(PYTORCH_MOBILE_OBSERVER)
  if (observer) {
    observer->onExit();
  }
#endif
  return result;
}

Function* Module::find_method(const std::string& basename) const {
  for (auto& fn : cu_->methods()) {
    if (fn->name() == basename) {
      return fn.get();
    }
  }
  return nullptr;
}

} // namespace mobile
} // namespace torch
} // namespace jit
