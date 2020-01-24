"""
To run this file by hand from the root of the PyTorch
repository, run:

python -m tools.autograd.gen_autograd \
       build/aten/src/ATen/Declarations.yaml \
       $OUTPUT_DIR \
       tools/autograd

Where $OUTPUT_DIR is where you would like the files to be
generated.  In the full build system, OUTPUT_DIR is
torch/csrc/autograd/generated/
"""

# gen_autograd.py generates C++ autograd functions and Python bindings.
#
# It delegates to the following scripts:
#
#  gen_autograd_functions.py: generates subclasses of torch::autograd::Node
#  gen_variable_type.py: generates VariableType.h which contains all tensor methods
#  gen_python_functions.py: generates Python bindings to THPVariable
#

import argparse
import copy
import os
import yaml
import re
from collections import defaultdict
from .utils import YamlLoader, split_name_params

# See NOTE [ Autograd View Variables ] in variable.h for details.
# A map: function name => name of the argument that all outputs are view of
VIEW_FUNCTIONS = {
    'numpy_T': 'self',
    'alias': 'self',
    'as_strided': 'self',
    'diagonal': 'self',
    'expand': 'self',
    'split': 'self',
    'split_with_sizes': 'self',
    'permute': 'self',
    'select': 'self',
    'slice': 'self',
    'squeeze': 'self',
    't': 'self',
    'transpose': 'self',
    'unfold': 'self',
    'unsqueeze': 'self',
    'view': 'self',
    'unbind': 'self',
    '_indices': 'self',
    '_values': 'self',
    'indices': 'self',
    'values': 'self',
    # sparse_coo ctor output should really be views of both indices and values,
    # but we only supports making as view of a single variable, and indices is
    # discrete anyways.
    # FIXME: clone indices on construction.
    'sparse_coo_tensor_with_dims_and_tensors': 'values',
}

# note: some VIEW_FUNCTIONS are just compositions of the view functions above
# this list contains both the root view functions and any that are purely composed
# of viewing functions, and is used by the JIT to determine when an operator
# returns a view of its inputs
RETURNS_VIEWS_OF_INPUT = set(VIEW_FUNCTIONS.keys()).union({'chunk', 'narrow'})

# note: For functions that return multiple views, we make a distinction between the
# functions that only perform viewing (called pure views here) versus all others (like
# user-defined Functions). This list is used to know which functions are pure views.
# A function should NOT be added to this list if:
# - Your backward function does more than just the backward of the views. For example if your
# function makes changes to the layout both in forward and backward.
# A function should be added to this list if:
# - You want to allow inplace modification of the result.
# - You are aware that inplace modification of the output will change the memory usage of the
#   backward as the multiple Nodes we create will force the gradient buffer to be created
#   for each Node, instead of once in the original multi-output Node.
# If you update this list, please update "test_multi_view_methods" in test_autograd.py
PURE_VIEW_FUNCTIONS = ['split', 'split_with_sizes', 'unbind']

def format_return_type(returns):
    if len(returns) == 0:
        return 'void'
    elif len(returns) == 1:
        return returns[0]['type']
    else:
        return_types = [r['type'] for r in returns]
        return 'std::tuple<{}>'.format(','.join(return_types))


def get_simple_type(arg):
    simple_type = arg['type']
    simple_type = simple_type.replace(' &', '').replace('const ', '')
    simple_type = simple_type.replace('Generator *', 'Generator')

    opt_match = re.match(r'c10::optional<(.+)>', simple_type)
    if opt_match:
        simple_type = '{}?'.format(opt_match.group(1))
    return simple_type


def load_aten_declarations(path):
    with open(path, 'r') as f:
        declarations = yaml.load(f, Loader=YamlLoader)

    # enrich declarations with additional information
    selected_declarations = []
    for declaration in declarations:
        if declaration.get('deprecated'):
            continue

        for arg in declaration['arguments']:
            arg['simple_type'] = get_simple_type(arg)
        for ret in declaration['returns']:
            ret['simple_type'] = get_simple_type(ret)

        declaration['formals'] = [arg['type'] + ' ' + arg['name']
                                  for arg in declaration['arguments']]
        declaration['args'] = [arg['name'] for arg in declaration['arguments']]
        declaration['type_method_formals'] = [arg['type'] + ' ' + arg['name']
                                              for arg in declaration['arguments']]
        declaration['type_method_args'] = [arg['name'] for arg in declaration['arguments']]
        declaration['api_name'] = declaration['name']
        declaration['return_type'] = format_return_type(declaration['returns'])

        declaration['base_name'] = declaration['name']
        selected_declarations.append(declaration)

    return selected_declarations


def load_deprecated_signatures(aten_decls, deprecated_path):
    def group_declarations_by_signature():
        d = defaultdict(list)
        for declaration in aten_decls:
            name = declaration['name']
            base_name = name[:-1] if declaration['inplace'] else name
            simple_types = [arg['simple_type'] for arg in declaration['arguments']]
            signature = '{}({})'.format(base_name, ', '.join(simple_types))
            d[signature].append(declaration)
        return d

    with open(deprecated_path, 'r') as f:
        deprecated_defs = yaml.load(f, Loader=YamlLoader)
    declarations = []
    declarations_by_signature = group_declarations_by_signature()

    def get_signature(name, params, call_args):
        # create a mapping of parameter name to parameter type
        types = dict([param.split(' ')[::-1] for param in params if param != '*'])
        # if the name in the call is not in the parameter list, assume it's
        # a literal Scalar
        rearranged_types = [types.get(arg, 'Scalar') for arg in call_args]
        return '{}({})'.format(name, ', '.join(rearranged_types))

    for deprecated in deprecated_defs:
        aten_name, call_args = split_name_params(deprecated['aten'])
        name, params = split_name_params(deprecated['name'])
        signature = get_signature(aten_name, params, call_args)

        for declaration in declarations_by_signature[signature]:
            declaration = copy.deepcopy(declaration)
            declaration['deprecated'] = True
            declaration['call_args'] = call_args

            call_arg_to_idx = {arg: i for i, arg in enumerate(call_args)}
            original_args = declaration['arguments']

            # Create an arguments list that uses the types from the original
            # ATen declaration, but the ordering and parameter names from
            # the deprecated overload. Any default parameter values from the
            # original ATen declaration are ignored.
            arguments = []
            kwarg_only = False
            for param in params:
                if param == '*':
                    kwarg_only = True
                    continue
                _, param_name = param.split(' ')
                original = original_args[call_arg_to_idx[param_name]]
                arguments.append({
                    'name': param_name,
                    'kwarg_only': kwarg_only,
                    'type': original['type'],
                    'simple_type': original['simple_type'],
                    'dynamic_type': original['dynamic_type'],
                    'output': original.get('output', False),
                })
            declaration['arguments'] = arguments
            declarations.append(declaration)
    return declarations


def gen_autograd(aten_path, out, autograd_dir, disable_autograd=False):
    aten_decls = load_aten_declarations(aten_path)

    # Parse and load derivatives.yaml
    from .load_derivatives import load_derivatives
    autograd_functions = load_derivatives(
        os.path.join(autograd_dir, 'derivatives.yaml'), aten_decls)

    template_path = os.path.join(autograd_dir, 'templates')

    # Generate VariableType.h/cpp
    if not disable_autograd:
        from .gen_variable_type import gen_variable_type
        gen_variable_type(out, aten_decls, template_path)

    # Generate Functions.h/cpp
    from .gen_autograd_functions import gen_autograd_functions_lib
    gen_autograd_functions_lib(
        out, autograd_functions, template_path)

    # Generate variable_factories.h
    from .gen_variable_factories import gen_variable_factories
    gen_variable_factories(
        out, aten_decls, template_path, disable_autograd=disable_autograd)


def gen_autograd_python(aten_path, out, autograd_dir):

    # TODO Deduplicate these four variable assignments

    aten_decls = load_aten_declarations(aten_path)

    # Parse and load derivatives.yaml
    from .load_derivatives import load_derivatives
    autograd_functions = load_derivatives(
        os.path.join(autograd_dir, 'derivatives.yaml'), aten_decls)

    template_path = os.path.join(autograd_dir, 'templates')

    # Load deprecated signatures
    deprecated = load_deprecated_signatures(
        aten_decls, os.path.join(autograd_dir, 'deprecated.yaml'))

    # Generate Functions.h/cpp
    from .gen_autograd_functions import gen_autograd_functions_python
    gen_autograd_functions_python(
        out, autograd_functions, template_path)

    # Generate Python bindings
    from . import gen_python_functions
    gen_python_functions.gen_py_variable_methods(
        out, aten_decls + deprecated, template_path)
    gen_python_functions.gen_py_torch_functions(
        out, aten_decls + deprecated, template_path)
    gen_python_functions.gen_py_nn_functions(
        out, aten_decls, template_path)


def main():
    parser = argparse.ArgumentParser(
        description='Generate autograd C++ files script')
    parser.add_argument('declarations', metavar='DECL',
                        help='path to Declarations.yaml')
    parser.add_argument('out', metavar='OUT',
                        help='path to output directory')
    parser.add_argument('autograd', metavar='AUTOGRAD',
                        help='path to autograd directory')
    args = parser.parse_args()
    gen_autograd(args.declarations, args.out, args.autograd)


if __name__ == '__main__':
    main()
