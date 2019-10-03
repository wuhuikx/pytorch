#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import unittest
from os import getenv
from unittest import mock

import torch
import torch.distributed as dist
import torch.distributed.rpc_backend_registry as rpc_backend_registry
from collections import namedtuple
from torch.distributed.internal_rpc_utils import _internal_rpc_pickler, PythonUDF

from common_utils import load_tests
from torch.distributed.rpc_api import RpcBackend
from dist_utils import dist_init

BACKEND = getenv("RPC_BACKEND", RpcBackend.PROCESS_GROUP)
RPC_INIT_URL = getenv("RPC_INIT_URL", "")


def stub_init_rpc_backend_handler(self_rank, self_name, init_method):
    return mock.Mock()  # RpcAgent.


# it is used to test python user defined function over rpc
# classes and functions are used to test python user defined class and
# methods over rpc
TensorClass = namedtuple("TensorClass", ["tensors"])


class MyPickleClass:
    def __init__(self):
        self.t = None

    def __getstate__(self):
        (pickled_python_udf, tensors) = _internal_rpc_pickler.serialize(
            PythonUDF(my_tensor_function,
                      (torch.ones(2, 2), torch.ones(2, 2)),
                      None))
        return (pickled_python_udf, tensors)

    def __setstate__(self, obj):
        python_udf = _internal_rpc_pickler.deserialize(obj[0], obj[1])
        result = python_udf.func(python_udf.args[0], python_udf.args[1])
        self.t = result

    def set(self, val):
        self.t = val


class MyClass:
    def __init__(self, a):
        self.a = a

    def my_instance_method(self, b):
        return self.a + b

    @classmethod
    def my_class_method(cls, d, e):
        return d + e

    @staticmethod
    def my_static_method(f):
        return f > 10


def run_nested_pickle(pickle_cls_instance, tensor):
    return pickle_cls_instance.t + tensor


def build_complex_tensors():
    a = torch.ones(3, 3)
    b = [a, a]
    c = [b, b]
    d = [a, b]
    e = {a : d}
    return [a, b, c, d, e]


def my_function(a, b, c):
    return a + b + c


def my_tensor_function(a, b):
    return a + b


def my_complex_tensor_function(list_input, tensor_class_input, dict_input):
    res = list_input[0]
    for t in list_input:
        res += t
    for k, v in dict_input.items():
        res += v
    complex_tensors = tensor_class_input.tensors
    return (res, complex_tensors[0], complex_tensors[1], complex_tensors[2])


def no_result():
    print("do nothing")


def nested_rpc(dst):
    return dist.rpc_sync(dst, torch.add, args=(torch.ones(2, 2), 1))

def multi_layer_nested_async_rpc(dst, world_size, ttl):
    # this method returns immediately without blocking the callee, but will
    # generate additional requests.
    if ttl > 0:
        current_dst = "worker{}".format(dst)
        next_dst = (dst + 1) % world_size
        dist.rpc(
            current_dst,
            multi_layer_nested_async_rpc,
            args=(
                next_dst,
                world_size,
                ttl - 1
            ),
            async_call=True
        )
        return 0

def light_rpc():
    return 0


def heavy_rpc(tensor):
    for i in range(1, 100):
        tensor *= i
        tensor /= i + 1
    return 0


def raise_func():
    raise ValueError("Expected error")


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests


@unittest.skipIf(
    sys.version_info < (3, 0),
    "Pytorch distributed rpc package " "does not support python2",
)
class RpcTest(object):
    @property
    def world_size(self):
        return 4

    @dist_init
    def test_worker_id(self):
        n = self.rank + 1
        peer_rank = n % self.world_size
        self_worker_id = dist.get_worker_id()
        peer_worker_id = dist.get_worker_id("worker{}".format(peer_rank))

        self.assertEqual(self_worker_id.name, "worker{}".format(self.rank))
        self.assertEqual(peer_worker_id.name, "worker{}".format(peer_rank))

        with self.assertRaisesRegex(RuntimeError, "Unknown destination worker"):
            unknown_worker_id = dist.get_worker_id("WorkerUnknown")

    @dist_init
    def test_self_add(self):
        self_worker_id = dist.get_worker_id()
        self_worker_name = "worker{}".format(self.rank)

        with self.assertRaisesRegex(
            RuntimeError, "does not support making RPC calls to self"
        ):
            dist.rpc_sync(self_worker_id, torch.add, args=(torch.ones(2, 2), 1))

        with self.assertRaisesRegex(
            RuntimeError, "does not support making RPC calls to self"
        ):
            dist.rpc_sync(self_worker_name, torch.add, args=(torch.ones(2, 2), 1))

    @mock.patch.object(torch.distributed.autograd, "_init")
    @mock.patch.object(torch.distributed.rpc_api, "init_rref_context")
    def test_register_rpc_backend_and_init_rpc_backend(
        self, mock_init_rref_context, mock_dist_autograd_init
    ):
        backend_name = "stub_backend"
        rpc_backend_registry.register_rpc_backend(
            backend_name, stub_init_rpc_backend_handler
        )
        dist.init_model_parallel(self_name="worker1", backend=backend_name, self_rank=1)

    @unittest.skipIf(
        BACKEND != RpcBackend.PROCESS_GROUP,
        "PROCESS_GROUP rpc backend specific test, skip",
    )
    def test_duplicate_name(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="gloo", rank=self.rank, world_size=self.world_size, store=store
        )
        with self.assertRaisesRegex(RuntimeError, "is not unique"):
            dist.init_model_parallel(
                self_name="duplicate_name",
                backend=BACKEND,
                self_rank=self.rank,
                init_method=RPC_INIT_URL,
            )
        dist.join_rpc()

    def test_reinit(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="gloo", rank=self.rank, world_size=self.world_size, store=store
        )
        dist.init_model_parallel(
            self_name="worker{}".format(self.rank),
            backend=BACKEND,
            self_rank=self.rank,
            init_method=RPC_INIT_URL,
        )
        with self.assertRaisesRegex(RuntimeError, "is already initialized"):
            dist.init_model_parallel(
                self_name="worker{}".format(self.rank),
                backend=BACKEND,
                self_rank=self.rank,
                init_method=RPC_INIT_URL,
            )
        dist.join_rpc()

    def test_init_invalid_backend(self):
        with self.assertRaisesRegex(RuntimeError, "Unrecognized RPC backend"):
            dist.init_model_parallel(
                self_name="worker{}".format(self.rank),
                backend="invalid",
                self_rank=self.rank,
                init_method=RPC_INIT_URL,
            )

    @unittest.skip("Test is flaky, see https://github.com/pytorch/pytorch/issues/25912")
    def test_invalid_names(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="gloo", rank=self.rank, world_size=self.world_size, store=store
        )

        with self.assertRaisesRegex(RuntimeError, "Worker name must match"):
            dist.init_model_parallel(self_name="abc*")

        with self.assertRaisesRegex(RuntimeError, "Worker name must match"):
            dist.init_model_parallel(self_name=" ")

        with self.assertRaisesRegex(RuntimeError, "must be non-empty"):
            dist.init_model_parallel(self_name="")

        # If the number in the message does not match, it is likely that the
        # value of MAX_NAME_LEN in RPC WorkerId has changed.
        with self.assertRaisesRegex(RuntimeError, "shorter than 128"):
            dist.init_model_parallel(
                self_name="".join(["a" for _ in range(500)]),
                backend=BACKEND,
                self_rank=self.rank,
                init_method=RPC_INIT_URL,
            )
        dist.join_rpc()

    @dist_init
    def test_add(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = dist.rpc_sync(
            "worker{}".format(dst_rank),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )
        self.assertEqual(ret, torch.ones(n, n) * 2)

    @dist_init
    def test_add_with_id(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        workder_id = dist.get_worker_id("worker{}".format(dst_rank))

        ret = dist.rpc_sync(
            workder_id, torch.add, args=(torch.ones(n, n), torch.ones(n, n))
        )
        self.assertEqual(ret, torch.ones(n, n) * 2)

    @dist_init
    def test_scalar_add(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = dist.rpc_sync(
            "worker{}".format(dst_rank), torch.add, args=(torch.ones(n, n), n)
        )
        self.assertEqual(ret, (torch.ones(n, n) + n))

    @dist_init
    def test_async_add(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        fut = dist.rpc_async(
            "worker{}".format(dst_rank),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )
        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)

    @dist_init
    def test_nonzero(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        x = torch.ones(self.world_size, self.world_size)
        x[self.rank][self.rank] = 0
        ret = dist.rpc_sync("worker{}".format(dst_rank), torch.nonzero, args=(x,))
        self.assertEqual(ret, x.nonzero())

    @dist_init
    def test_multi_rpc(self):
        dst_rank = (self.rank + 1) % self.world_size
        for i in range(20):
            n = i + self.rank + 1
            ret = dist.rpc_sync(
                "worker{}".format(dst_rank),
                torch.add,
                args=(torch.ones(n, n), torch.ones(n, n)),
            )
            self.assertEqual(ret, torch.ones(n, n) * 2)

    @dist_init
    def test_sync_rpc(self):
        dst_rank = (self.rank + 1) % self.world_size
        for i in range(20):
            dist.sync_rpc()
            n = i + self.rank + 1
            ret1 = dist.rpc_sync(
                "worker{}".format(dst_rank),
                torch.add,
                args=(torch.ones(n, n), torch.ones(n, n)),
            )
            dist.sync_rpc()
            ret2 = dist.rpc_sync(
                "worker{}".format(dst_rank), torch.add, args=(torch.ones(n, n), 2)
            )
            dist.sync_rpc()
            self.assertEqual(ret1, torch.ones(n, n) * 2)
            self.assertEqual(ret2, torch.ones(n, n) * 3)

    @dist_init
    def test_join_rpc(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = dist.rpc_sync(
            "worker{}".format(dst_rank),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )
        self.assertEqual(ret, torch.ones(n, n) * 2)
        dist.join_rpc()

        with self.assertRaisesRegex(RuntimeError, "^RPC has not been initialized"):
            dist.rpc_sync(
                "worker{}".format(dst_rank),
                torch.add,
                args=(torch.ones(n, n), torch.ones(n, n)),
            )

        # it's safe to call join_rpc() multiple times
        dist.join_rpc()

    @dist_init
    def test_py_built_in(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = dist.rpc_sync("worker{}".format(dst_rank), min, args=(n, n + 1, n + 2))
        self.assertEqual(ret, min(n, n + 1, n + 2))

    @dist_init
    def test_py_user_defined(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = dist.rpc_sync(
            "worker{}".format(dst_rank),
            my_function,
            kwargs={"a": n, "b": n + 1, "c": n + 2},
        )
        self.assertEqual(ret, my_function(n, n + 1, n + 2))

    @dist_init
    def test_py_class_constructor(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = dist.rpc_sync("worker{}".format(dst_rank), MyClass, args=(n,))
        self.assertEqual(ret.a, n)

    @dist_init
    def test_py_class_instance_method(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = dist.rpc_sync(
            "worker{}".format(dst_rank), MyClass(2).my_instance_method, args=(n,)
        )
        self.assertEqual(ret, MyClass(2).my_instance_method(n))

    @dist_init
    def test_py_class_method(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = dist.rpc_sync(
            "worker{}".format(dst_rank), MyClass.my_class_method, args=(n, n + 1)
        )
        self.assertEqual(ret, MyClass.my_class_method(n, n + 1))

    @dist_init
    def test_py_class_static_method(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = dist.rpc_sync(
            "worker{}".format(dst_rank), MyClass.my_static_method, args=(n + 10,)
        )
        self.assertEqual(ret, MyClass.my_static_method(n + 10))

    @dist_init
    def test_py_multi_async_call(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        dst_worker_id = dist.get_worker_id("worker{}".format(dst_rank))
        fut1 = dist.rpc_async(dst_worker_id, MyClass.my_static_method, args=(n + 10,))
        fut2 = dist.rpc_async(dst_worker_id, min, args=(n, n + 1, n + 2))
        self.assertEqual(fut1.wait(), MyClass.my_static_method(n + 10))
        self.assertEqual(fut2.wait(), min(n, n + 1, n + 2))

    @dist_init
    def test_py_no_return_result(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = dist.rpc_sync("worker{}".format(dst_rank), no_result)
        self.assertEqual(ret, no_result())

    @dist_init
    def test_py_tensors(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = dist.rpc("worker{}".format(dst_rank),
                       my_tensor_function,
                       args=(torch.ones(n, n), torch.ones(n, n)))
        self.assertEqual(ret,
                         my_tensor_function(torch.ones(n, n),
                                            torch.ones(n, n)))

    @dist_init
    def test_py_tensors_multi_async_call(self):
        futs = []
        n = self.rank + 1
        dst_rank = n % self.world_size
        for i in range(100):
            fut = dist.rpc("worker{}".format(dst_rank),
                           my_tensor_function,
                           args=(torch.ones(i, i), torch.ones(i, i)),
                           async_call=True)
            futs.append(fut)

        j = 0
        for fut in futs:
            self.assertEqual(fut.wait(),
                             my_tensor_function(torch.ones(j, j),
                                                torch.ones(j, j)))
            j += 1

    @dist_init
    def test_py_tensors_in_container(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        a = [torch.ones(n, n), torch.ones(n, n)]
        b = TensorClass(build_complex_tensors())
        c = {"foo": torch.ones(n, n), "bar": torch.ones(n, n)}
        ret = dist.rpc("worker{}".format(dst_rank),
                       my_complex_tensor_function,
                       args=(a, b, c))
        self.assertEqual(ret, my_complex_tensor_function(a, b, c))

    @dist_init
    def test_py_nested_pickle(self):
        n = self.rank + 1
        dst_rank = n % self.world_size

        ret = dist.rpc("worker{}".format(dst_rank),
                       run_nested_pickle,
                       args=(MyPickleClass(), torch.ones(2, 2)))

        m = MyPickleClass()
        m.set(my_tensor_function(torch.ones(2, 2), torch.ones(2, 2)))
        self.assertEqual(ret, run_nested_pickle(m, torch.ones(2, 2)))

    @dist_init
    def test_py_function_exception(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        with self.assertRaisesRegex(Exception, "TypeError"):
            ret = dist.rpc_sync("worker{}".format(dst_rank), no_result, args=(10,))

    @dist_init
    def test_py_raise_in_user_func(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        fut = dist.rpc_async("worker{}".format(dst_rank), raise_func)
        with self.assertRaisesRegex(Exception, "ValueError"):
            fut.wait()

    @dist_init
    def test_nested_rpc(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = dist.rpc_sync(
            "worker{}".format(dst_rank),
            nested_rpc,
            args=("worker{}".format(self.rank),),
        )
        self.assertEqual(ret, torch.ones(2, 2) + 1)

    def _stress_test_rpc(self, f, repeat=1000, args=()):
        import time

        n = self.rank + 1
        dst_rank = n % self.world_size
        futs = []
        tik = time.time()
        for _ in range(repeat):
            fut = dist.rpc_async("worker{}".format(dst_rank), f, args=args)
            futs.append(fut)

        for fut in futs:
            self.assertEqual(fut.wait(), 0)
        tok = time.time()
        print(
            "Rank {} finished testing {} {} times in {} seconds.".format(
                self.rank, f.__name__, repeat, tok - tik
            )
        )

    @dist_init
    def test_stress_light_rpc(self):
        self._stress_test_rpc(light_rpc)

    @dist_init
    def test_stress_heavy_rpc(self):
        self._stress_test_rpc(heavy_rpc, repeat=20, args=(torch.ones(100, 100),))

    @dist_init
    def test_builtin_remote_ret(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref = dist.remote(
            "worker{}".format(dst_rank),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )
        self.assertEqual(rref.to_here(), torch.ones(n, n) * 2)

    @dist_init
    def test_multi_builtin_remote_ret(self):
        m = 10
        n = self.rank + 1
        dst_rank = n % self.world_size
        rrefs = []
        expected = []
        for i in range(m):
            n = n + i
            rrefs.append(
                dist.remote(
                    "worker{}".format(dst_rank),
                    torch.add,
                    args=(torch.ones(n, n), torch.ones(n, n)),
                )
            )
            expected.append(torch.ones(n, n) * 2)

        for i in range(m):
            self.assertEqual(rrefs[i].to_here(), expected[i])

    @dist_init
    def test_multi_layer_nested_async_rpc(self):
        # This test will exit right away, but there will be a chain of async
        # RPCs. The termination algorithm should detect those messages properly.
        # Otherwise, some peer could exit early, leaving others to timeout
        # errors or connection closed errors.
        ttl = 20
        n = self.rank + 1
        dst_rank = n % self.world_size

        multi_layer_nested_async_rpc(dst_rank, self.world_size, ttl)
