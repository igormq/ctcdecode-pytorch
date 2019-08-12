import argparse
import glob
import os
import platform
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

project_version = '0.1'

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_processes",
                    default=1,
                    type=int,
                    help="Number of cpu processes to build package. (default: %(default)d)")
args = parser.parse_known_args()

# reconstruct sys.argv to pass to setup below
sys.argv = [sys.argv[0]] + args[1]

extra_compile_args = ['-O3']
libraries = []

if os.environ.get('DEBUG', '0') == '1':
    extra_compile_args = ['-O0', '-g']

extra_compile_args += [
    '-DNDEBUG', '-DKENLM_MAX_ORDER=20', '-Wno-unused-local-typedefs', '-Wno-sign-compare', '-std=c++11', '-w',
    '-DINCLUDE_KENLM'
]

if platform.system() == 'Linux':
    libraries += ['stdc++', 'rt']
elif platform.system() == 'Darwin':
    extra_compile_args += ["-stdlib=libc++", "-mmacosx-version-min=10.9"]
    libraries += ['c++']

include_dirs = [
    '../../third_party/kenlm', '../../third_party/openfst/src/include', '../../third_party/ThreadPool',
    '../../third_party/boost'
]

sources = (glob.glob('../../third_party/kenlm/util/*.cc') + glob.glob('../../third_party/kenlm/lm/*.cc') +
           glob.glob('../../third_party/kenlm/util/double-conversion/*.cc'))

sources += glob.glob('../../third_party/openfst/src/lib/*.cc')

sources = [fn for fn in sources if not (fn.endswith('main.cc') or fn.endswith('test.cc') or fn.endswith('unittest.cc'))]

sources = glob.glob('*.cpp')

decoder_module = CppExtension(name='_C',
                              sources=sources,
                              language='c++',
                              include_dirs=include_dirs + ['./'],
                              extra_compile_args=extra_compile_args,
                              libraries=libraries)

setup(name='ds_ctcdecoder',
      version=project_version,
      description="""DS CTC decoder""",
      ext_modules=[decoder_module],
      cmdclass={'build_ext': BuildExtension})
