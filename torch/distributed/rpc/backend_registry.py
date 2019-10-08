from __future__ import absolute_import, division, print_function, unicode_literals


_BACKEND_REGISTRY = {}


def _get_backend_registry():
    return _BACKEND_REGISTRY


def is_backend_registered(backend_name):
    return backend_name in _get_backend_registry()


def register_backend(backend_name, init_backend_handler):
    """Registers a new RPC backend.

    Arguments:
        backend (str): backend string to identify the handler.
        handler (function): Handler that is invoked when the
            `_init_rpc()` function is called with a backend.
             This returns the agent.
    """
    backend_registry = _get_backend_registry()
    if backend_name in backend_registry:
        raise RuntimeError("RPC backend {}: already registered".format(backend_name))
    backend_registry[backend_name] = init_backend_handler


def init_backend(backend_name, *args, **kwargs):
    backend_registry = _get_backend_registry()
    if backend_name not in backend_registry:
        raise RuntimeError("No rpc_init handler for {}.".format(backend_name))
    return backend_registry[backend_name](*args, **kwargs)
