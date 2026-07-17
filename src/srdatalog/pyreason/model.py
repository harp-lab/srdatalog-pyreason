'''Program-agnostic source and result types for the PyReason frontend.'''

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceClause:
  predicate: str
  terms: tuple[str, ...]
  lower: float
  upper: float


@dataclass(frozen=True)
class SourceRule:
  text: str
  name: str
  head_predicate: str
  head_terms: tuple[str, ...]
  head_annotation: str | None
  head_lower: float
  head_upper: float
  delay: int
  clauses: tuple[SourceClause, ...]
  weights: tuple[float, ...] = ()
  infer_edges: bool = False
  set_static: bool = False


@dataclass(frozen=True)
class SourceFact:
  text: str
  name: str
  predicate: str
  arguments: tuple[str, ...]
  lower: float
  upper: float
  start_time: int
  end_time: int
  static: bool


@dataclass(frozen=True)
class SourceProgram:
  '''Neutral capture of the public PyReason program state.

  Annotation callables remain first-class Python objects.  The compatibility
  compiler derives their meaning from the callable and rule metadata; users do
  not provide an SRDatalog-specific semantic duplicate.
  '''

  rules: tuple[SourceRule, ...]
  facts: tuple[SourceFact, ...]
  graphml_path: str | None
  closed_world_predicates: frozenset[str]
  annotation_functions: tuple[tuple[str, object], ...]
  settings: tuple[tuple[str, bool], ...] = ()


@dataclass(frozen=True)
class TemporalIntervalOutput:
  '''How a native uint32 relation maps back to PyReason temporal atoms.'''

  relation: str
  predicate: str
  argument_columns: tuple[int, ...]
  time_column: int
  lower_column: int
  upper_column: int

  @property
  def required_columns(self) -> tuple[int, ...]:
    return tuple(
      sorted(
        {
          *self.argument_columns,
          self.time_column,
          self.lower_column,
          self.upper_column,
        }
      )
    )


@dataclass(frozen=True)
class NativePlan:
  program: Any
  project_name: str
  data_dir: Path
  outputs: tuple[TemporalIntervalOutput, ...]
  symbol_names: dict[int, str]
  rewriter: str


@dataclass(frozen=True)
class TemporalIntervalRow:
  predicate: str
  arguments: tuple[str, ...]
  time: int
  lower: float
  upper: float


@dataclass(frozen=True)
class ExecutionResult:
  rows: tuple[TemporalIntervalRow, ...]
  backend: str
  rewriter: str
  metadata: dict[str, object]
