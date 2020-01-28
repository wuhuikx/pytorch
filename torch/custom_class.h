
#pragma once

#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/stack.h>
#include <c10/util/C++17.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/TypeList.h>
#include <c10/util/TypeTraits.h>
#include <torch/csrc/jit/custom_class.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/script/compilation_unit.h>
#include <torch/csrc/jit/tracer.h>
#include <torch/csrc/utils/variadic.h>
#include <iostream>
#include <sstream>


namespace torch {
namespace jit {

TORCH_API std::vector<c10::RegisterOperators>& registeredOps();
TORCH_API std::shared_ptr<script::CompilationUnit>& classCU();

namespace detail {
template <class R, class...>
struct types {
  using type = types;
};
template <class Sig>
struct args;
template <class R, class CurClass, class... Args>
struct args<R (CurClass::*)(Args...)> : types<R, Args...> {};
template <class R, class CurClass, class... Args>
struct args<R (CurClass::*)(Args...) const> : types<R, Args...> {};
template <class Sig>
using args_t = typename args<Sig>::type;
} // namespace detail
template <class... Types>
detail::types<void, Types...> init() { return detail::types<void, Types...>{}; }

// To bind custom classes into Torchscript, use an API very similar to Pybind's.
// Currently exposes one class `torch::jit::class_<T>` and 2 methods.
// - Constructing `torch::jit::class_<Foo>` registers `Foo` in Python and
// Torchscript, and puts it under `torch.classes.Foo` in Python.
// - torch::jit::class_<Foo>.def("method1", &Foo::method1) does some template
// metaprogramming to introspect the function types and register the operator
// for use in Torchscript.
// - torch::jit::class_<Foo>.def(torch::jit::init<int64_t, int64_t>()) registers
// the Foo(int, int) constructor.
// see test/custom_operator/classes.cpp and
// test/custom_operator/test_custom_classes.py for example usages

template <class CurClass>
class class_ {
  static_assert(std::is_base_of<CustomClassHolder, CurClass>::value,
    "torch::jit::class_<T> requires T to inherit from CustomClassHolder");

  std::string className;
  std::string qualClassName;
  ClassTypePtr classTypePtr;

  const std::string parentModule = "classes";
  const std::string topModule = "__torch__.torch";

 public:
  class_(std::string className_) : className(std::move(className_)) {
    // Currently we register everything as a python class just for convenience.
    // We'll want to remove this at some point to get rid of the python
    // dependency. It would require significant changes to class registration,
    // (I think)?
    qualClassName = topModule + "." + parentModule + "." + className;

    // We currently represent custom classes as torchscript classes with a
    // capsule attribute
    classTypePtr =
        ClassType::create(c10::QualifiedName(qualClassName), classCU());
    classTypePtr->addAttribute("capsule", CapsuleType::get());

    c10::getCustomClassTypeMap().insert({typeid(c10::intrusive_ptr<CurClass>).name(),
                              c10::StrongTypePtr(classCU(), classTypePtr)});
    c10::getCustomClassTypeMap().insert({typeid(c10::tagged_capsule<CurClass>).name(),
                              c10::StrongTypePtr(classCU(), classTypePtr)});

    classCU()->register_type(classTypePtr);
  }

  template <typename... Types>
  class_& def(detail::types<void, Types...>) { // Used in combination with
                                               // torch::jit::init<...>()
    auto func = [](c10::tagged_capsule<CurClass> self, Types... args) {
      auto classObj = c10::make_intrusive<CurClass>(args...);
      auto genericPtr = c10::static_intrusive_pointer_cast<torch::jit::CustomClassHolder>(classObj);
      auto capsule = IValue(genericPtr);
      auto object = self.ivalue.toObject();
      object->setSlot(0, capsule);
    };

    defineMethod<void>("__init__", std::move(func));
    return *this;
  }
  template <
      typename Method,
      std::enable_if_t<
          std::is_member_function_pointer<std::decay_t<Method>>::value,
          bool> = false>
  class_& def(std::string name, Method&& m) {
    auto res = def_(
        std::move(name),
        std::forward<Method>(m),
        detail::args_t<std::remove_reference_t<decltype(m)>>{});
    return *this;
  }
  template <
      typename Func,
      std::enable_if_t<
          !std::is_member_function_pointer<std::decay_t<Func>>::value,
          bool> = false>
  class_& def(std::string name, Func&& f) {
    auto res = def_(
        std::move(name),
        std::forward<Func>(f),
        detail::args_t<std::remove_reference_t<decltype(&Func::operator())>>{});
    return *this;
  }

 private:
  template<typename R, typename Func>
  void defineMethod(std::string name, Func func) {
    auto graph = std::make_shared<Graph>();
    auto qualFuncName = className + "::" + name;
    ensure_c10_registerer_defined();
    registeredOps().push_back(
        torch::RegisterOperators().op(qualFuncName, std::move(func)));
    auto func_symbol = c10::Symbol::fromQualString(qualFuncName);
    auto ops = torch::jit::getAllOperatorsFor(func_symbol);
    TORCH_CHECK(ops.size() == 1);
    auto &schema = ops[0]->schema();

    for (const auto& arg : schema.arguments()) {
      graph->addInput()->setType(arg.type());
    }

    auto opCall = graph->insertNode(graph->create(
        func_symbol, graph->inputs(), schema.returns().size()));
    Value* res;
    if (schema.returns().size() > 1) {
      const auto& returns = schema.returns();
      size_t op_invocation_idx = 0;
      for (const auto& ret : returns) {
        opCall->output(op_invocation_idx++)->setType(ret.type());
      }
      res = graph->insertNode(graph->createTuple(opCall->outputs()))->output();
    } else if (schema.returns().size() == 1) {
      const auto& returns = schema.returns();
      res = opCall->output()->setType(returns[0].type());
    } else {
      res = graph->insertConstant(IValue())->setType(NoneType::get());
    }
    graph->registerOutput(res);

    auto method = classCU()->create_function(qualClassName + "." + name, graph);
    classTypePtr->addMethod(method);
  }

  template <
      typename Func,
      typename R,
      typename... Types,
      std::enable_if_t<
          std::is_member_function_pointer<std::decay_t<Func>>::value,
          bool> = false>
  class_& def_(std::string name, Func f, detail::types<R, Types...> funcInfo) {
    auto func = [f = std::move(f)](
                    c10::intrusive_ptr<CurClass> cur, Types... args) {
      return at::guts::invoke(f, *cur, args...);
    };
    defineMethod<R>(std::move(name), std::move(func));
    return *this;
  }

  template <typename R, typename Head, typename... Tail>
  void assert_self_type(detail::types<R, Head, Tail...> funcInfo) {
    static_assert(
        std::is_same<std::decay_t<Head>, c10::intrusive_ptr<CurClass>>::value,
        "First argument of a registered lambda method must be an intrusive_ptr<> of the corresponding class.");
  }

  template <
      typename Func,
      typename R,
      typename... Types,
      std::enable_if_t<
          !std::is_member_function_pointer<std::decay_t<Func>>::value,
          bool> = false>
  class_& def_(
      std::string name,
      Func&& f,
      detail::types<R, Types...> funcInfo) {
    assert_self_type(funcInfo);
    defineMethod<R>(std::move(name), std::forward<Func>(f));
    return *this;
  }
};

} // namespace jit

} // namespace torch
