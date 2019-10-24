from __future__ import absolute_import, division, print_function, unicode_literals

import threading
from functools import wraps
from os import getenv

import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch.distributed.rpc.api import RpcBackend


if not dist.is_available():
    print("c10d not available, skipping tests")
    sys.exit(0)


class TestConfig:
    __slots__ = ["rpc_backend"]

    def __init__(self, *args, **kwargs):
        assert len(args) == 0, "TestConfig only takes kwargs."
        for k, v in kwargs.items():
            setattr(self, k, v)


TEST_CONFIG = TestConfig(rpc_backend=getenv("RPC_BACKEND", RpcBackend.PROCESS_GROUP))
INIT_METHOD_TEMPLATE = "file://{file_name}"


MASTER_RANK = 0
_ALL_NODE_NAMES = set()
_DONE_NODE_NAMES = set()
_TERMINATION_SIGNAL = threading.Event()


def on_master_follower_report_done(worker_name):
    assert (
        worker_name in _ALL_NODE_NAMES
    ), "{worker_name} is not expected by master.".format(worker_name=worker_name)
    assert (
        worker_name not in _DONE_NODE_NAMES
    ), "{worker_name} report done twice.".format(worker_name=worker_name)
    _DONE_NODE_NAMES.add(worker_name)
    if _ALL_NODE_NAMES != _DONE_NODE_NAMES:
        return
    set_termination_signal()


def set_termination_signal():
    assert not _TERMINATION_SIGNAL.is_set(), "Termination signal got set twice."
    _TERMINATION_SIGNAL.set()


def dist_init(setup_model_parallel=True):
    assert isinstance(setup_model_parallel, bool), (
        "setup_model_parallel must be a bool value"
    )

    def decorator(old_test_method):
        """
        We use this decorator for setting up and tearing down state since
        MultiProcessTestCase runs each `test*` method in a separate process and
        each process just runs the `test*` method without actually calling
        'setUp' and 'tearDown' methods of unittest.
        """

        @wraps(old_test_method)
        def new_test_method(self, *arg, **kwargs):
            self.worker_id = self.rank
            self.worker_name_to_id = {
                "worker{}".format(rank): rank for rank in range(self.world_size)
            }

            if setup_model_parallel:
                global _ALL_NODE_NAMES
                _ALL_NODE_NAMES = self.worker_name_to_id.keys()

                dist.init_process_group(
                    backend="gloo",
                    init_method=self.init_method,
                    rank=self.rank,
                    world_size=self.world_size,
                )
                # Use enough 'num_send_recv_threads' until we fix https://github.com/pytorch/pytorch/issues/26359
                rpc.init_model_parallel(
                    self_name="worker%d" % self.rank,
                    backend=TEST_CONFIG.rpc_backend,
                    init_method=self.init_method,
                    self_rank=self.rank,
                    worker_name_to_id=self.worker_name_to_id,
                    num_send_recv_threads=16,
                )

            ret = old_test_method(self, *arg, **kwargs)

            if setup_model_parallel:
                # Follower reports done.
                if self.rank == MASTER_RANK:
                    on_master_follower_report_done("worker{}".format(MASTER_RANK))
                else:
                    rpc.rpc_async(
                        "worker{}".format(MASTER_RANK),
                        on_master_follower_report_done,
                        args=("worker{}".format(self.rank),),
                    )

                # Master waits for followers to report done.
                # Follower waits for master's termination command.
                _TERMINATION_SIGNAL.wait()
                if self.rank == MASTER_RANK:
                    # Master sends termination command.
                    futs = []
                    for dst_rank in range(self.world_size):
                        # torch.distributed.rpc module does not support sending to self.
                        if dst_rank == MASTER_RANK:
                            continue
                        dst_name = "worker{}".format(dst_rank)
                        fut = rpc.rpc_async(dst_name, set_termination_signal, args=())
                        futs.append(fut)
                    for fut in futs:
                        assert fut.wait() is None, "Sending termination signal failed."

                # Close RPC.
                rpc.join_rpc()

            return ret

        return new_test_method

    return decorator
