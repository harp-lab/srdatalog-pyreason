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
  clause_source_positions: tuple[int, ...] = ()


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
  graph_attribute: bool = False


@dataclass(frozen=True)
class SourceProgram:
  '''Neutral capture of the public PyReason program state.

  Annotation callables remain first-class Python objects.  The compatibility
  compiler derives their meaning from the callable and rule metadata; users do
  not provide an SRDatalog-specific semantic duplicate. Node and edge domains
  include graph and fact-created components, with logical birth times kept
  separately so a future fact cannot enlarge an earlier grounding domain.
  Inconsistent predicate pairs preserve the public consistency declarations.
  '''

  rules: tuple[SourceRule, ...]
  facts: tuple[SourceFact, ...]
  graphml_path: str | None
  closed_world_predicates: frozenset[str]
  annotation_functions: tuple[tuple[str, object], ...]
  settings: tuple[tuple[str, bool], ...] = ()
  node_domain: tuple[str, ...] = ()
  edge_domain: tuple[tuple[str, str], ...] = ()
  node_birth_times: tuple[tuple[str, int], ...] = ()
  edge_birth_times: tuple[tuple[str, str, int], ...] = ()
  inconsistent_predicates: tuple[tuple[str, str], ...] = ()
  reorder_clauses_node_first: bool | None = None


@dataclass(frozen=True)
class TemporalIntervalOutput:
  '''How a native uint32 relation maps back to PyReason temporal atoms.'''

  relation: str
  predicate: str
  argument_columns: tuple[int, ...]
  time_column: int
  lower_column: int
  upper_column: int
  suppress_bottom: bool = False

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
class RuleWitnessOutput:
  '''How an instrumented rule-candidate relation maps to logical witnesses.

  These columns identify a source-rule grounding and the interval versions it
  observed.  They are ordinary set-valued tuples, not physical row IDs and not
  semiring annotations attached to the materialized world relation.
  '''

  relation: str
  rule_index: int
  rule_name: str
  head_predicate: str
  head_argument_columns: tuple[int, ...]
  head_time_column: int
  rank_column: int | None
  candidate_lower_column: int
  candidate_upper_column: int
  binding_columns: tuple[tuple[str, int], ...]
  body_lower_columns: tuple[int, ...]
  body_upper_columns: tuple[int, ...]
  closed_world_clause_positions: tuple[int, ...] = ()

  @property
  def required_columns(self) -> tuple[int, ...]:
    rank_columns = () if self.rank_column is None else (self.rank_column,)
    return tuple(
      sorted(
        {
          *self.head_argument_columns,
          self.head_time_column,
          *rank_columns,
          self.candidate_lower_column,
          self.candidate_upper_column,
          *(column for _, column in self.binding_columns),
          *self.body_lower_columns,
          *self.body_upper_columns,
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
  witness_outputs: tuple[RuleWitnessOutput, ...] = ()
  effective_source: SourceProgram | None = None
  live_time_relation: str | None = None
  callback_fd_relations: tuple[str, ...] = ()
  callback_read_predicates: tuple[str, ...] = ()


@dataclass(frozen=True)
class TemporalIntervalRow:
  predicate: str
  arguments: tuple[str, ...]
  time: int
  lower: float
  upper: float
  inconsistent: bool = False
  raw_lower: float | None = None
  raw_upper: float | None = None
  frozen: bool = False


@dataclass(frozen=True)
class RuleCandidateRow:
  '''One historical source-rule grounding retained by an on-demand replay.'''

  rule_index: int
  rule_name: str
  head_predicate: str
  head_arguments: tuple[str, ...]
  head_time: int
  rank: int | None
  candidate_lower: float
  candidate_upper: float
  bindings: tuple[tuple[str, str], ...]
  body_intervals: tuple[tuple[float, float], ...]
  closed_world_clause_positions: tuple[int, ...] = ()

  @property
  def witness_key(self) -> tuple[object, ...]:
    return (
      self.rule_index,
      self.head_predicate,
      self.head_arguments,
      self.head_time,
      self.rank,
      self.candidate_lower,
      self.candidate_upper,
      self.bindings,
      self.body_intervals,
    )


@dataclass(frozen=True)
class ExecutionResult:
  rows: tuple[TemporalIntervalRow, ...]
  backend: str
  rewriter: str
  metadata: dict[str, object]
  witness_rows: tuple[RuleCandidateRow, ...] = ()
