#
# Copyright 2016, 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of CLgen.
#
# CLgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CLgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CLgen.  If not, see <http://www.gnu.org/licenses/>.
#
import os
import sys

import pytest
from absl import app
from absl import logging

import deeplearning.clgen.errors
from deeplearning.clgen import errors
from deeplearning.clgen import preprocess
from deeplearning.clgen.tests import testlib as tests


# Invoke tests with UPDATE_GS_FILES set to update the gold standard
# tests. E.g.:
#
#   $ UPDATE_GS_FILES=1 python3 ./setup.py test
#
UPDATE_GS_FILES = True if 'UPDATE_GS_FILES' in os.environ else False


def preprocess_pair(basename, preprocessor=preprocess.preprocess):
  gs_path = tests.data_path(os.path.join('cl', str(basename) + '.gs'),
                            exists=not UPDATE_GS_FILES)
  tin_path = tests.data_path(os.path.join('cl', str(basename) + '.cl'))

  # Run preprocess
  tin = tests.data_str(tin_path)
  tout = preprocessor(tin)

  if UPDATE_GS_FILES:
    gs = tout
    with open(gs_path, 'w') as outfile:
      outfile.write(gs)
      print("\n-> updated gold standard file '{}' ...".format(gs_path),
            file=sys.stderr, end=' ')
  else:
    gs = tests.data_str(gs_path)

  return (gs, tout)


@pytest.mark.skip(reason="TODO(cec)")
def test_preprocess():
  assert len(set(preprocess_pair('sample-1'))) == 1


def test_compile_cl_bytecode_good_code():
  """Test that bytecode is produced for good code."""
  assert preprocess.compile_cl_bytecode("kernel void A(global float* a) {}",
                                        "<anon>", use_shim=False)


def test_compile_cl_bytecode_undefined_type():
  """Test that error is raised when kernel contains undefined type."""
  with pytest.raises(errors.ClangException):
    preprocess.compile_cl_bytecode("kernel void A(global FLOAT_T* a) {}",
                                   "<anon>", use_shim=False)


def test_compile_cl_bytecode_shim_type():
  """Test that bytecode is produced for code with shim type."""
  assert preprocess.compile_cl_bytecode("kernel void A(global FLOAT_T* a) {}",
                                        "<anon>", use_shim=True)


@pytest.mark.skip(reason="TODO(cec)")
def test_bytecode_features_empty_code():
  # Generated by compile_cl_bytecode from: 'kernel void A(global float* a) {}'.
  bc = """\
; ModuleID = '-'
source_filename = "-"
target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-nvcl"

; Function Attrs: noinline norecurse nounwind readnone
define spir_kernel void @A(float addrspace(1)* nocapture) local_unnamed_addr 
#0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 
!kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
  ret void
}

attributes #0 = { noinline norecurse nounwind readnone 
"correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" 
"less-precise-fpmad"="false" "no-frame-pointer-elim"="true" 
"no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" 
"no-jump-tables"="false" "no-nans-fp-math"="false" 
"no-signed-zeros-fp-math"="false" "no-trapping-math"="false" 
"stack-protector-buffer-size"="8" "target-features"="-satom" 
"unsafe-fp-math"="false" "use-soft-float"="false" }

!nvvm.annotations = !{!0}
!llvm.module.flags = !{!1}
!opencl.ocl.version = !{!2}
!llvm.ident = !{!3}

!0 = !{void (float addrspace(1)*)* @A, !"kernel", i32 1}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 1, i32 0}
!3 = !{!"clang version 5.0.0 (tags/RELEASE_500/final)"}
!4 = !{i32 1}
!5 = !{!"none"}
!6 = !{!"float*"}
!7 = !{!""}
"""
  assert preprocess.bytecode_features(bc, "<anon>")


def test_compiler_preprocess_cl_no_change():
  """Test that code without preprocessor directives is unchanged."""
  src = "kernel void A(global int*a ) {}"
  assert preprocess.compiler_preprocess_cl(src) == src


def test_compiler_preprocess_cl_whitespace():
  """Test that preprocessor output produces exactly one terminating newling."""
  src = "kernel void A(global int*a ) {}"
  # Leading whitespace is stripped.
  assert preprocess.compiler_preprocess_cl('\n\n' + src) == src
  # Trailing whitespace is stripped.
  assert preprocess.compiler_preprocess_cl(src + '\n\n') == src


def test_compiler_preprocess_cl_user_directives():
  """Test inlining of user-defined preprocessor directives."""
  src = """\
#define MY_TYPE int
kernel void A(global MY_TYPE* a) {}
#ifdef SOME_CONDITIONAL_THING
kernel void B() {}
#endif
"""
  out = "kernel void A(global int* a) {}"
  assert preprocess.compiler_preprocess_cl(src) == out


def test_compiler_preprocess_cl_undefined_macro():
  """Test that code with undefined macro is unchanged."""
  src = "kernel void A(global MY_TYPE* a) {}"
  assert preprocess.compiler_preprocess_cl(src) == src


def test_rewriter_good_code():
  """Test that OpenCL rewriter renames variables and functions."""
  rewritten = preprocess.rewrite_cl("""\
__kernel void FOOBAR(__global int * b) {
    if (  b < *b) {
          *b *= 2;
    }
}\
""")
  assert rewritten == """\
__kernel void A(__global int * a) {
    if (  a < *a) {
          *a *= 2;
    }
}\
"""


@pytest.mark.skip(reason="TODO(cec)")
def test_preprocess_shim():
  """Test that code which contains defs in opencl-shim can compile."""
  # FLOAT_T is defined in shim header. Preprocess will fail if FLOAT_T is
  # undefined.
  with pytest.raises(errors.BadCodeException):
    preprocess.preprocess("""
__kernel void A(__global FLOAT_T* a) { int b; }""", use_shim=False)

  assert preprocess.preprocess("""
__kernel void A(__global FLOAT_T* a) { int b; }""", use_shim=True)


@pytest.mark.skip(reason="TODO(cec)")
def test_ugly_preprocessed():
  # empty kernel protoype is rejected
  with pytest.raises(errors.NoCodeException):
    preprocess.preprocess("""\
__kernel void A() {
}\
""")
  # kernel containing some code returns the same.
  assert """\
__kernel void A() {
  int a;
}\
""" == preprocess.preprocess("""\
__kernel void A() {
  int a;
}\
""")


@pytest.mark.skip(reason="TODO(cec)")
def test_preprocess_stable():
  code = """\
__kernel void A(__global float* a) {
  int b;
  float c;
  int d = get_global_id(0);

  a[d] *= 2.0f;
}"""
  # pre-processing is "stable" if the code doesn't change
  out = code
  for _ in range(5):
    out = preprocess.preprocess(out)
    assert out == code


@tests.needs_linux  # FIXME: GPUVerify support on macOS.
def test_gpuverify():
  code = """\
__kernel void A(__global float* a) {
  int b = get_global_id(0);
  a[b] *= 2.0f;
}"""
  assert preprocess.gpuverify(code,
                              ["--local_size=64", "--num_groups=128"]) == code


@tests.needs_linux  # FIXME: GPUVerify support on macOS.
def test_gpuverify_data_race():
  code = """\
__kernel void A(__global float* a) {
  a[0] +=  1.0f;
}"""
  with pytest.raises(deeplearning.clgen.errors.GPUVerifyException):
    preprocess.gpuverify(code, ["--local_size=64", "--num_groups=128"])


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized command line flags.')
  logging.set_verbosity(logging.DEBUG)
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  app.run(main)
