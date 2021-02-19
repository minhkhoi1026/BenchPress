# Copyright (c) 2016-2020 Chris Cummins.
#
# clgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# clgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with clgen.  If not, see <https://www.gnu.org/licenses/>.
"""This file contains utility code for working with clang.

This module does not expose any preprocessor functions for CLgen. It contains
wrappers around Clang binaries, which preprocessor functions can use to
implement specific behavior. See deeplearning.clgen.preprocessors.cxx.Compile()
for an example.
"""
import json
import re
import humanize
import subprocess
import tempfile
import typing
import string
import clang.cindex
from absl import flags
from deeplearning.clgen.util import environment
from eupy.native import  logger as l

# The marker used to mark stdin from clang pre-processor output.
CLANG_STDIN_MARKER = re.compile(r'# \d+ "<stdin>" 2')
# Options to pass to clang-format.
# See: http://clang.llvm.org/docs/ClangFormatStyleOptions.html
CLANG_FORMAT_CONFIG = {
  "BasedOnStyle": "Google",
  "ColumnLimit": 5000,
  "IndentWidth": 2,
  "AllowShortBlocksOnASingleLine": False,
  "AllowShortCaseLabelsOnASingleLine": False,
  "AllowShortFunctionsOnASingleLine": False,
  "AllowShortLoopsOnASingleLine": False,
  "AllowShortIfStatementsOnASingleLine": False,
  "DerivePointerAlignment": False,
  "PointerAlignment": "Left",
  "BreakAfterJavaFieldAnnotations": True,
  "BreakBeforeInheritanceComma": False,
  "BreakBeforeTernaryOperators": False,
  "AlwaysBreakAfterReturnType": "None",
  "AlwaysBreakAfterDefinitionReturnType": "None",
}
clang.cindex.Config.set_library_path(environment.LLVM_LIB)
if environment.LLVM_VERSION == 9:
  # LLVM 9 needs libclang explicitly defined.
  clang.cindex.Config.set_library_file(environment.LLVM_LIB + "/libclang.so.9")

CLANG = environment.CLANG
CLANG_FORMAT = environment.CLANG_FORMAT

def StripPreprocessorLines(src: str) -> str:
  """Strip preprocessor remnants from clang frontend output.

  Args:
    src: Clang frontend output.

  Returns:
    The output with preprocessor output stripped.
  """
  lines = src.split("\n")
  # Determine when the final included file ends.
  for i in range(len(lines) - 1, -1, -1):
    if CLANG_STDIN_MARKER.match(lines[i]):
      break
  else:
    return ""
  # Strip lines beginning with '#' (that's preprocessor stuff):
  return "\n".join([line for line in lines[i:] if not line.startswith("#")])


def Preprocess(
  src: str,
  cflags: typing.List[str],
  timeout_seconds: int = 60,
  strip_preprocessor_lines: bool = True,
):
  """Run input code through the compiler frontend to inline macros.

  This uses the repository clang binary.

  Args:
    src: The source code to preprocess.
    cflags: A list of flags to be passed to clang.
    timeout_seconds: The number of seconds to allow before killing clang.
    strip_preprocessor_lines: Whether to strip the extra lines introduced by
      the preprocessor.

  Returns:
    The preprocessed code.

  Raises:
    ClangException: In case of an error.
    ClangTimeout: If clang does not complete before timeout_seconds.
  """
  cmd = [
    "timeout",
    "-s9",
    str(timeout_seconds),
    str(CLANG),
    "-E",
    "-c",
    "-",
    "-o",
    "-",
  ] + cflags

  process = subprocess.Popen(
    cmd,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    universal_newlines=True,
  )
  stdout, stderr = process.communicate(src)
  if process.returncode == 9:
    raise ValueError(
      f"Clang preprocessor timed out after {timeout_seconds}s"
    )
  elif process.returncode != 0:
    raise ValueError(stderr)
  if strip_preprocessor_lines:
    return StripPreprocessorLines(stdout)
  else:
    return stdout

def ProcessCompileLlvmBytecode(
  src: str, suffix: str, cflags: typing.List[str], timeout_seconds: int = 60
) -> str:
  """Compile input code into textual LLVM byte code using clang system binary.

  Args:
    src: The source code to compile.
    suffix: The suffix to append to the source code temporary file. E.g. '.c'
      for a C program.
    cflags: A list of flags to be passed to clang.
    timeout_seconds: The number of seconds to allow before killing clang.

  Returns:
    The textual LLVM byte code.

  Raises:
    ValueError: In case of an error.
    ValueError: If clang does not complete before timeout_seconds.
  """
  builtin_cflags = ["-S", "-emit-llvm", "-o", "-"]
  with tempfile.NamedTemporaryFile(
    "w", prefix="phd_deeplearning_clgen_preprocessors_clang_", suffix=suffix
  ) as f:
    f.write(src)
    f.flush()
    cmd = (
      ["timeout", "-s9", str(timeout_seconds), str(CLANG), f.name]
      + builtin_cflags
      + cflags
    )
    process = subprocess.Popen(
      cmd,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      universal_newlines=True,
    )
    stdout, stderr = process.communicate()
  if process.returncode == 9:
    raise ValueError(f"Clang timed out after {timeout_seconds}s")
  elif process.returncode != 0:
    raise ValueError("/*\n{}\n*/\n{}".format(stderr, src))
  return stdout

def CompileLlvmBytecode(src: str,
                        suffix: str,
                        cflags: typing.List[str],
                        return_diagnostics: bool = False,
                        ) -> str:
  """Compile input code into textual LLVM byte code using clang.Cindex python module.

  Args:
    src: The source code to compile.
    suffix: The suffix to append to the source code temporary file. E.g. '.c'
      for a C program.
    cflags: A list of flags to be passed to clang.

  Returns:
    The textual LLVM byte code.

  Raises:
    ValueError: In case of an error.
  """
  builtin_cflags = ["-S", "-emit-llvm", "-o", "-"]
  with tempfile.NamedTemporaryFile(
    "w", prefix="phd_deeplearning_clgen_preprocessors_clang_", suffix=suffix
  ) as f:
    f.write(src)
    f.flush()
    try:
      unit = clang.cindex.TranslationUnit.from_source(f.name, args = builtin_cflags + cflags)
    except clang.cindex.TranslationUnitLoadError as e:
      raise ValueError(e)
    diagnostics = [str(d) for d in unit.diagnostics if d.severity > 2]
    locations   = [(d.location.line, d.location.column) for d in unit.diagnostics if d.severity > 2]
    if len(diagnostics) > 0:
      if return_diagnostics:
        return src, locations
      else:
        raise ValueError("/*\n{}\n*/\n{}".format('\n'.join(diagnostics), src))
    else:
      if return_diagnostics:
        return src, []
      else:
        return src

def ClangFormat(text: str, suffix: str, timeout_seconds: int = 60) -> str:
  """Run clang-format on a source to enforce code style.

  Args:
    text: The source code to run through clang-format.
    suffix: The suffix to append to the source code temporary file. E.g. '.c'
      for a C program.
    timeout_seconds: The number of seconds to allow clang-format to run for.

  Returns:
    The output of clang-format.

  Raises:
    ClangFormatException: In case of an error.
    ClangTimeout: If clang-format does not complete before timeout_seconds.
  """

  cmd = [
    "timeout",
    "-s9",
    str(timeout_seconds),
    str(CLANG_FORMAT),
    "-assume-filename",
    f"input{suffix}",
    "-style={}".format(json.dumps(CLANG_FORMAT_CONFIG))
  ]
  process = subprocess.Popen(
    cmd,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    universal_newlines=True,
  )
  stdout, stderr = process.communicate(text)
  if process.returncode == 9:
    raise ValueError(f"clang-format timed out after {timeout_seconds}s")
  elif process.returncode != 0:
    raise ValueError(stderr)
  return stdout

def DeriveSourceVocab(src: str,
                      token_list: typing.Set[str],
                      suffix: str,
                      cflags: typing.List[str],
                      ) -> typing.Dict[str, str]:
  """Pass source code through clang's lexer and return set of tokens.

  Args:
    src: The source code to compile.
    token_list: External set of grammar tokens for target language.
    suffix: The suffix to append to the source code temporary file. E.g. '.c'
      for a C program.
    cflags: A list of flags to be passed to clang.

  Returns:
    Set of unique source code tokens

  Raises:
    ValueError: In case of an error.
  """
  builtin_cflags = ["-S", "-emit-llvm", "-o", "-"]
  with tempfile.NamedTemporaryFile(
    "w", prefix="phd_deeplearning_clgen_preprocessors_clang_", suffix=suffix
  ) as f:
    f.write(src)
    f.flush()
    try:
      unit = clang.cindex.TranslationUnit.from_source(f.name, args = builtin_cflags + cflags)
    except clang.cindex.TranslationUnitLoadError as e:
      raise ValueError(e)

    tokens = {}
    for ch in string.printable:
      # Store all printable characters as char-based, to save time iterating literals.
      tokens["{}-char-based".format(ch)] = ''
    for idx, t in enumerate(unit.get_tokens(extent = unit.cursor.extent)):
      str_t = str(t.spelling)
      if str_t in token_list or t.kind in {clang.cindex.TokenKind.KEYWORD, clang.cindex.TokenKind.PUNCTUATION}:
        tokens[str_t] = ' '
      else:
        if t.kind != clang.cindex.TokenKind.LITERAL and clang.cindex.Cursor.from_location(unit, t.extent.end).kind not in {clang.cindex.CursorKind.CALL_EXPR}:
          tokens[str_t] = ' '

    return tokens

def AtomizeSource(src: str,
                  vocab: typing.Set[str],
                  suffix: str,
                  cflags: typing.List[str],
                  ) -> typing.List[str]:
  """
  Split source code into token atoms with clang's lexer.

  Args:
    src: The source code to compile.
    vocab: Optional set of learned vocabulary of tokenizer.
    suffix: The suffix to append to the source code temporary file. E.g. '.c'
      for a C program.
    cflags: A list of flags to be passed to clang.

  Returns:
    Source code as a list of tokens.

  Raises:
    ValueError: In case of an error.
  """
  builtin_cflags = ["-S", "-emit-llvm", "-o", "-"]
  with tempfile.NamedTemporaryFile(
    "w", prefix="phd_deeplearning_clgen_preprocessors_clang_", suffix=suffix
  ) as f:
    f.write(src)
    f.flush()
    try:
      unit = clang.cindex.TranslationUnit.from_source(f.name, args = builtin_cflags + cflags)
    except clang.cindex.TranslationUnitLoadError as e:
      raise ValueError(e)
    tokens = []
    for idx, t in enumerate(unit.get_tokens(extent = unit.cursor.extent)):
      str_t = t.spelling
      if str_t in vocab:
        tokens.append(str(t.spelling))
      else:
        for ch in str_t:
          tokens.append("{}-char-based".format(ch))
    return tokens
