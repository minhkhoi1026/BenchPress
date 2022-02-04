"""
Evaluation script for clsmith mutation program.
"""
import typing
import tempfile
import subprocess
import multiprocessing
import pathlib
import json
import datetime
import sqlite3
import functools
import os
import tqdm
import math

import sqlalchemy as sql
from sqlalchemy.ext import declarative
from absl import flags

from deeplearning.clgen.features import extractor
from deeplearning.clgen.preprocessors import opencl
from deeplearning.clgen.preprocessors import c
from deeplearning.clgen.util import plotter
from deeplearning.clgen.util import environment
from deeplearning.clgen.util import crypto
from deeplearning.clgen.util import sqlutil
from deeplearning.clgen.util import logging as l
from deeplearning.clgen.experiments import public

FLAGS = flags.FLAGS
CLSMITH         = environment.CLSMITH
CLSMITH_INCLUDE = environment.CLSMITH

Base = declarative.declarative_base()

class CLSmithSample(Base, sqlutil.ProtoBackedMixin):
  """A database row representing a CLgen sample.

  This is the clgen.CLSmithSample protocol buffer in SQL format.
  """
  __tablename__    = "clsmith_samples"
  # entry id
  id                     : int = sql.Column(sql.Integer,    primary_key = True)
  # unique hash of sample text
  sha256                 : str = sql.Column(sql.String(64), nullable = False, index = True)
  # String-format generated kernel
  sample                 : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # String-format generated header file
  include                : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # encoded sample text
  encoded_sample         : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Whether the generated sample compiles or not.
  compile_status         : bool = sql.Column(sql.Boolean,  nullable = False)
  # CLSmithSample's vector of features.
  feature_vector         : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Length of total sequence in number of tokens
  num_tokens             : int = sql.Column(sql.Integer,   nullable = False)
  # Date
  date_added             : datetime.datetime = sql.Column(sql.DateTime, nullable=False)

  @classmethod
  def FromArgs(cls,
               id      : int,
               sample  : str,
               include : str,
               encoded_sample : str,
               compile_status : bool,
               feature_vector : str,
               num_tokens     : int,
               ) -> "CLSmithSample":
    """
    Do you want to use CLSmithDatabase as a means to store only code
    without much fuss ? This function is for you!
    """
    return CLSmithSample(**{
      "id"             : id,
      "sha256"         : crypto.sha256_str(sample),
      "sample"         : sample,
      "include"        : include,
      "encoded_sample" : encoded_sample,
      "compile_status" : compile_status,
      "feature_vector" : feature_vector,
      "num_tokens"     : num_tokens,
      "date_added"     : datetime.datetime.utcnow(),
    })

class CLSmithDatabase(sqlutil.Database):
  """A database of CLgen samples."""

  def __init__(self, url: str, must_exist: bool = False):
    super(CLSmithDatabase, self).__init__(url, Base, must_exist = must_exist)

  @property
  def count(self):
    """Number of samples in DB."""
    with self.Session() as s:
      count = s.query(CLSmithSample).count()
    return count

def execute_clsmith(idx: int, tokenizer, timeout_seconds: int = 15) -> typing.List[CLSmithSample]:
  """
  Execute clsmith and return sample.
  """
  try:
    tdir = pathlib.Path(FLAGS.local_filesystem).resolve()
  except Exception:
    tdir = None

  extra_args = ["-I{}".format(CLSMITH_INCLUDE)]
  with tempfile.NamedTemporaryFile("w", prefix = "clsmith_", suffix = ".cl", dir = tdir) as f:
    cmd =[
      "timeout",
      "-s9",
      str(timeout_seconds),
      CLSMITH,
      "-o",
      str(f.name)
    ]
    process = subprocess.Popen(
      cmd,
      stdout = subprocess.PIPE,
      stderr = subprocess.PIPE,
      universal_newlines = True,
    )
    try:
      stdout, stderr = process.communicate()
    except TimeoutError:
      return None

    contentfile = open(str(f.name), 'r').read()

  try:
    ks = opencl.ExtractSingleKernelsHeaders(
           opencl.StripDoubleUnderscorePrefixes(
             opencl.ClangPreprocess(
               c.StripIncludes(contentfile),
               extra_args = extra_args,
             )
           )
         )
  except ValueError as e:
    l.logger().error(contentfile)
    raise e

  samples = []
  for kernel, include in ks:
    sample = opencl.SequentialNormalizeIdentifiers(kernel, extra_args = extra_args)
    encoded_sample = tokenizer.AtomizeString(sample)
    try:
      stdout = opencl.Compile(sample, header_file = include, extra_args = extra_args)
      compile_status = True
    except ValueError as e:
      stdout = str(e)
      compile_status = False

    samples.append(
      CLSmithSample.FromArgs(
        id             = idx,
        sample         = stdout,
        include        = include,
        encoded_sample = ','.join(encoded_sample),
        compile_status = compile_status,
        feature_vector = extractor.ExtractRawFeatures(sample, header_file = include, extra_args = extra_args),
        num_tokens     = len(encoded_sample)
      )
    )
  return samples

@public.evaluator
def GenerateCLSmith(**kwargs) -> None:
  """
  Compare mutec mutation tool on github's database against BenchPress.
  Comparison is similar to KAverageScore comparison.
  """
  clsmith_path = kwargs.get('clsmith_path', '')
  tokenizer    = kwargs.get('tokenizer')

  if not pathlib.Path(CLSMITH).exists():
    raise FileNotFoundError("CLSmith executable not found: {}".format(CLSMITH))

  # Initialize clsmith database
  clsmith_db = CLSmithDatabase(url = "sqlite:///{}".format(str(pathlib.Path(clsmith_path).resolve())), must_exist = False)

  while True:
    chunk_size = 1000
    f = functools.partial(execute_clsmith, tokenizer = tokenizer, timeout_seconds = 15)
    pool = multiprocessing.Pool()
    try:
      entries = []
      for samples in tqdm.tqdm(pool.imap_unordered(f, range(chunk_size)), total = chunk_size, desc = "Generate CLSmith Samples", leave = False):
        if samples:
          for sample in samples:
            entries.append(sample)

      db_idx = clsmith_db.count
      with clsmith_db.Session(commit = True) as s:
        for entry in entries:
          exists = s.query(CLSmithSample.sha256).filter_by(sha256 = entry.sha256).scalar() is not None
          if not exists:
            entry.id = db_idx
            s.add(entry)
            db_idx += 1
        s.commit()
    except KeyboardInterrupt as e:
      pool.terminate()
      break
    except Exception as e:
      l.logger().error(e)
      pool.terminate()
      raise e
    pool.close()
  return
