"""
Feature Extraction module for Autophase paper features.
"""
import subprocess
import tempfile
import typing

from deeplearning.clgen.preprocessors import opencl
from deeplearning.clgen.util import environment

from eupy.hermes import client

AUTOPHASE = ["-load", environment.AUTOPHASE, "-autophase"]

class AutophaseFeatures(object):
  """
  TODO write description.
  """
  def __init__(self):
    return

  @classmethod
  def ExtractFeatures(cls, src: str, use_aux_headers: bool = True) -> typing.Dict[str, float]:
    return cls.RawToDictFeats(cls.ExtractRawFeatures(src, use_aux_headers = use_aux_headers))

  @classmethod
  def ExtractRawFeatures(cls, src: str, use_aux_headers: bool = True) -> str:
    return opencl.CompileOptimizer(src, AUTOPHASE, use_aux_headers = use_aux_headers)

  @classmethod
  def RawToDictFeats(cls, str_feats: str) -> typing.Dict[str, float]:
    return {feat.split(' : ')[0]: int(feat.split(' : ')[1]) for feat in str_feats.split('\n') if ' : ' in feat}
