from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

class get_pybind_include(object):
    """Helper class to defer importing pybind11 until it is actually installed."""
    def __str__(self):
        return pybind11.get_include()

ext_modules = [
    Extension(
        'src_cascades.simulator_backend_cpp',
        ['src/src_cascades/core/simulator_backend.cpp'],
        include_dirs=[
            get_pybind_include(),
        ],
        language='c++'
    ),
]

# As in https://pybind11.readthedocs.io/en/stable/compiling.html#automatic-unique-variable-names
# for MSVC support
class BuildExt(build_ext):
    def build_extensions(self):
        c = self.compiler.compiler_type
        opts = []
        if c == 'msvc':
            opts.append('/EHsc')
            opts.append('/std:c++17')
        else:
            opts.append('-std=c++17')
        
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)

setup(
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)