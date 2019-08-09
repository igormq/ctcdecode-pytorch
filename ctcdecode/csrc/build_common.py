#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import glob
import os
import platform
import shlex
import subprocess
import sys
from multiprocessing.dummy import Pool

from torch.utils.cpp_extension import include_paths


def build_common(common_files,
                 includes,
                 args,
                 libs,
                 out_name='common.a',
                 build_dir='temp_build/temp_build',
                 num_parallel=1):
    compiler = os.environ.get('CXX', 'g++')
    ar = os.environ.get('AR', 'ar')
    libtool = os.environ.get('LIBTOOL', 'libtool')
    cflags = os.environ.get('CFLAGS', '') + os.environ.get('CXXFLAGS', '')

    for file in common_files:
        outfile = os.path.join(build_dir, os.path.splitext(file)[0] + '.o')
        outdir = os.path.dirname(outfile)
        if not os.path.exists(outdir):
            print('mkdir', outdir)
            os.makedirs(outdir)

    def build_one(file):
        outfile = os.path.join(build_dir, os.path.splitext(file)[0] + '.o')
        if os.path.exists(outfile):
            return

        cmd = '{cc} -fPIC -c {cflags} {args} {includes} {infile} -o {outfile} {libs}'.format(
            cc=compiler,
            cflags=cflags,
            args=' '.join(args),
            includes=' '.join('-I' + i for i in includes + include_paths()),
            infile=file,
            libs=' '.join('-l' + l for l in libs),
            outfile=outfile,
        )
        print(cmd)
        subprocess.check_call(shlex.split(cmd))
        return outfile

    pool = Pool(num_parallel)
    obj_files = list(pool.imap_unordered(build_one, common_files))

    if sys.platform.startswith('darwin'):
        cmd = '{libtool} -static -o {outfile} {infiles}'.format(
            libtool=libtool,
            outfile=out_name,
            infiles=' '.join(obj_files),
        )
        print(cmd)
        subprocess.check_call(shlex.split(cmd))
    else:
        cmd = '{ar} rcs {outfile} {infiles}'.format(ar=ar, outfile=out_name, infiles=' '.join(obj_files))
        print(cmd)
        subprocess.check_call(shlex.split(cmd))