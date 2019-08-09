import glob
import os
import sys
import platform

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

from build_common import build_common

extra_compile_args = ['-O3']
if os.environ.get('DEBUG', '0') == '1':
    extra_compile_args = ['-O0', '-g']
extra_compile_args += [
    '-DNDEBUG', '-DKENLM_MAX_ORDER=20', '-Wno-unused-local-typedefs', '-Wno-sign-compare', '-std=c++11', '-w',
    '-DINCLUDE_KENLM'
]

libraries = []
if platform.system() == 'Linux':
    libraries += ['stdc++', 'rt']
elif platform.system() == 'Darwin':
    extra_compile_args += ["-stdlib=libc++", "-mmacosx-version-min=10.9"]
    libraries += ['c++']

include_dirs = [
    '../../third_party/kenlm', '../../third_party/openfst/src/include', '../../third_party/ThreadPool',
    '../../third_party/boost'
]

common_sources = (glob.glob('../../third_party/kenlm/util/*.cc') + glob.glob('../../third_party/kenlm/lm/*.cc') +
                  glob.glob('../../third_party/kenlm/util/double-conversion/*.cc'))
common_sources += glob.glob('../../third_party/openfst/src/lib/*.cc')
common_sources = [
    fn for fn in common_sources if not (fn.endswith('main.cc') or fn.endswith('test.cc') or fn.endswith('unittest.cc'))
]

project_version = '0.1'

build_dir = 'temp_build/temp_build'
common_build = 'common.a'
if not os.path.exists(common_build):
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)

    build_common(common_sources, include_dirs, extra_compile_args, libraries, out_name='common.a', build_dir=build_dir)

decoder_module = CppExtension(name='_C',
                              sources=glob.glob('*.cpp'),
                              language='c++',
                              include_dirs=include_dirs + ['./'],
                              extra_compile_args=extra_compile_args,
                              libraries=libraries,
                              extra_link_args=[common_build])

setup(name='ds_ctcdecoder',
      version=project_version,
      description="""DS CTC decoder""",
      ext_modules=[decoder_module],
      cmdclass={'build_ext': BuildExtension})
