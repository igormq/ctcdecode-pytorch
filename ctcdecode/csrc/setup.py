#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

from setuptools import setup, Extension, distutils

from torch.utils.cpp_extension import BuildExtension, CppExtension
import glob
import argparse
import multiprocessing.pool
import os
import platform
import sys

from build_common import *

IS_WINDOWS = (platform.system() == 'Windows')
IS_DARWIN = (platform.system() == 'Darwin')
IS_LINUX = (platform.system() == 'Linux')

if IS_LINUX:
    ext_libs = ['stdc++', 'rt']
elif IS_DARWIN:
    ext_libs = ['c++']
    # ext_libs = []
    ARGS += ["-stdlib=libc++", "-mmacosx-version-min=10.9"]
else:
    ext_libs = []

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_processes",
                    default=1,
                    type=int,
                    help="Number of cpu processes to build package. (default: %(default)d)")
args = parser.parse_known_args()

# reconstruct sys.argv to pass to setup below
sys.argv = [sys.argv[0]] + args[1]

project_version = '0.1'

build_dir = 'temp_build/temp_build'
common_build = 'common.a'

if not os.path.exists(common_build):
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)

    build_common(out_name='common.a', build_dir=build_dir, num_parallel=args[0].num_processes)

decoder_module = CppExtension(name='_C',
                              sources=glob.glob('*.cpp'),
                              language='c++',
                              include_dirs=INCLUDES,
                              extra_compile_args=ARGS,
                              libraries=ext_libs,
                              extra_link_args=[common_build])

setup(name='ds_ctcdecoder',
      version=project_version,
      description="""DS CTC decoder""",
      ext_modules=[decoder_module],
      cmdclass={'build_ext': BuildExtension})
