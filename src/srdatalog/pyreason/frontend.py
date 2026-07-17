'''Extensible compiler boundary for captured PyReason programs.'''

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from .model import ExecutionResult, NativePlan, SourceProgram


class RewriteRejected(NotImplementedError):
  '''A registered rewriter claimed the source program but cannot preserve it.'''


class AnnotationRewriter(Protocol):
  name: str

  def claims(self, source: SourceProgram) -> bool: ...

  def rewrite(
    self,
    source: SourceProgram,
    *,
    timesteps: int,
    output_root: str | Path,
  ) -> NativePlan: ...


@dataclass(frozen=True)
class LazyAnnotationRewriter:
  '''Delay importing an application compiler plugin until rewrite selection.'''

  module: str
  attribute: str

  @property
  def name(self) -> str:
    return f'{self.module}:{self.attribute}'

  def _rewriter(self) -> AnnotationRewriter:
    return cast(
      AnnotationRewriter,
      getattr(importlib.import_module(self.module), self.attribute),
    )

  def claims(self, source: SourceProgram) -> bool:
    return self._rewriter().claims(source)

  def rewrite(
    self,
    source: SourceProgram,
    *,
    timesteps: int,
    output_root: str | Path,
  ) -> NativePlan:
    return self._rewriter().rewrite(
      source,
      timesteps=timesteps,
      output_root=output_root,
    )


_REWRITER_ATTRIBUTE = '__srdatalog_annotation_rewriter__'


def register_annotation_rewriter(
  annotation_function: object,
  rewriter: AnnotationRewriter,
) -> None:
  '''Attach a compiler rule directly to a first-class annotation callable.'''
  existing = getattr(annotation_function, _REWRITER_ATTRIBUTE, None)
  if existing is not None and existing != rewriter:
    raise ValueError('annotation function already has a different SRDatalog rewriter')
  try:
    setattr(annotation_function, _REWRITER_ATTRIBUTE, rewriter)
  except (AttributeError, TypeError) as exc:
    raise TypeError('annotation callable cannot carry an SRDatalog rewriter') from exc


def _source_rewriters(source: SourceProgram) -> tuple[AnnotationRewriter, ...]:
  found: list[AnnotationRewriter] = []
  for _, function in source.annotation_functions:
    rewriter = getattr(function, _REWRITER_ATTRIBUTE, None)
    if rewriter is not None and not any(existing is rewriter for existing in found):
      found.append(rewriter)
  return tuple(found)


def try_rewrite(
  source: SourceProgram,
  *,
  timesteps: int,
  output_root: str | Path,
) -> NativePlan | None:
  '''Apply the rewriter carried by a registered source annotation, if any.'''
  for rewriter in _source_rewriters(source):
    if rewriter.claims(source):
      return rewriter.rewrite(source, timesteps=timesteps, output_root=output_root)
  return None


def try_execute(
  source: SourceProgram,
  *,
  timesteps: int,
  output_root: str | Path,
  cache_base: str | Path,
  jobs: int = 8,
  timeout: float = 600,
) -> ExecutionResult | None:
  plan = try_rewrite(source, timesteps=timesteps, output_root=output_root)
  if plan is None:
    return None
  from .runtime import execute

  return execute(plan, cache_base=cache_base, jobs=jobs, timeout=timeout)
