'''Demand-driven provenance planning for captured PyReason programs.

The ordinary materialization does not carry derivation witnesses.  This module
builds a separate, query-directed backward slice after a caller asks for an
explanation.  Its values are compiler data, not eagerly accumulated semiring
annotations: a demand selects source rules, rule demands select body demands,
and matching source facts terminate a branch.

The rewrite deliberately retains source-rule identity.  Two rules that derive
the same temporal atom therefore remain two possible witnesses even when their
heads have the same predicate and arguments.
'''

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias

from .model import SourceProgram, SourceRule, TemporalIntervalRow


@dataclass(frozen=True)
class DemandSeed:
  '''A grounded atom or one logical evidence occurrence of that atom.

  The optional observed endpoints identify the interval version read by a
  parent rule.  This is stable semantic identity for demand replay, not a
  physical row address.  Closed-world defaults use an explicit mode because
  their stored range ``[0,1]`` was read operationally as false.
  '''

  predicate: str
  arguments: tuple[str, ...]
  time: int
  observed_lower: float | None = None
  observed_upper: float | None = None
  evidence_mode: Literal['materialized', 'closed-world-default'] = 'materialized'

  def __post_init__(self) -> None:
    if self.time < 0:
      raise ValueError('a demand seed cannot have a negative time')
    if (self.observed_lower is None) != (self.observed_upper is None):
      raise ValueError('an evidence occurrence requires both observed interval endpoints')
    if self.evidence_mode == 'closed-world-default':
      assert self.observed_lower is not None
      assert self.observed_upper is not None
      if (self.observed_lower, self.observed_upper) != (
        0.0,
        1.0,
      ) and self.observed_lower <= self.observed_upper:
        raise ValueError(
          'a closed-world default must record stored [0,1] or a crossed interval repaired to [0,1]'
        )

  @property
  def has_observed_interval(self) -> bool:
    return self.observed_lower is not None


@dataclass(frozen=True)
class DemandPattern:
  '''A magic predicate key; ``None`` is an unbound argument position.'''

  predicate: str
  arguments: tuple[str | None, ...]
  time: int


@dataclass(frozen=True)
class ConstantIntervalTransform:
  '''A rule head contributes the same interval for every grounding.'''

  lower: float
  upper: float


@dataclass(frozen=True)
class GroupedCallbackMarker:
  '''Marks a PyReason callback evaluated over rows grouped by the head key.'''

  annotation_name: str


@dataclass(frozen=True)
class GroupedCallbackTransform:
  '''A rule head interval is computed by its registered grouped callback.'''

  callback: GroupedCallbackMarker
  weights: tuple[float, ...]


IntervalTransform: TypeAlias = ConstantIntervalTransform | GroupedCallbackTransform


@dataclass(frozen=True)
class RuleWitnessIdentity:
  '''Stable identity of one source rule selected for one demanded head.'''

  rule_index: int
  rule_name: str
  demanded_head: DemandPattern


@dataclass(frozen=True)
class ClauseDemand:
  '''One body position in a selected source-rule witness.'''

  position: int
  predicate: str
  terms: tuple[str, ...]
  arguments: tuple[str | None, ...]
  time: int
  lower: float
  upper: float

  @property
  def demand(self) -> DemandPattern:
    return DemandPattern(self.predicate, self.arguments, self.time)


@dataclass(frozen=True)
class RuleWitness:
  '''Typed backward-rewrite record for one selected source rule.'''

  identity: RuleWitnessIdentity
  source_text: str
  head_predicate: str
  head_terms: tuple[str, ...]
  head_bindings: tuple[tuple[str, str], ...]
  body: tuple[ClauseDemand, ...]
  delay: int
  interval_transform: IntervalTransform
  infer_edges: bool
  set_static: bool


@dataclass(frozen=True)
class FactWitnessIdentity:
  '''Stable identity of a source-fact leaf at the demanded time.'''

  fact_index: int
  fact_name: str
  arguments: tuple[str, ...]
  time: int


@dataclass(frozen=True)
class FactLeaf:
  '''A backward slice stops at a matching source fact.'''

  identity: FactWitnessIdentity
  source_text: str
  predicate: str
  lower: float
  upper: float
  static: bool
  demanded_by: DemandPattern


@dataclass(frozen=True)
class DemandRewrite:
  '''Stable result of the magic-set/backward-slice planning pass.'''

  seeds: tuple[DemandSeed, ...]
  demands: tuple[DemandPattern, ...]
  rule_witnesses: tuple[RuleWitness, ...]
  fact_leaves: tuple[FactLeaf, ...]


@dataclass(frozen=True)
class GroundedRuleWitness:
  '''A one-hop rule witness with every source variable bound.'''

  rule: RuleWitness
  bindings: tuple[tuple[str, str], ...]


TraceEntity: TypeAlias = str | tuple[str, ...]


@dataclass(frozen=True)
class PyReasonClauseComponent:
  '''One ``Clause-N`` cell in PyReason's rule-trace presentation.'''

  position: int
  predicate: str
  entities: tuple[TraceEntity, ...]

  def as_trace_cell(self) -> list[TraceEntity]:
    '''Return the node/edge list shape used by PyReason trace dataframes.'''

    return list(self.entities)


def rewrite_for_demands(
  source: SourceProgram,
  seeds: Iterable[DemandSeed],
) -> DemandRewrite:
  '''Build the finite backward slice reachable from grounded demand seeds.

  This is a planning pass only.  It neither changes ordinary materialization
  nor records witnesses while materialization runs.  Unknown body variables
  are represented as unbound magic-key positions; processing each distinct
  key at most once makes zero-delay recursive programs terminate.
  '''

  stable_seeds = tuple(dict.fromkeys(seeds))
  pending: deque[DemandPattern] = deque(
    DemandPattern(seed.predicate, seed.arguments, seed.time) for seed in stable_seeds
  )
  seen: set[DemandPattern] = set()
  demands: list[DemandPattern] = []
  rule_witnesses: list[RuleWitness] = []
  fact_leaves: list[FactLeaf] = []

  while pending:
    demand = pending.popleft()
    if demand in seen:
      continue
    seen.add(demand)
    demands.append(demand)

    fact_leaves.extend(_matching_fact_leaves(source, demand))
    for rule_index, rule in enumerate(source.rules):
      witness = _select_rule(rule, rule_index, demand)
      if witness is None:
        continue
      rule_witnesses.append(witness)
      for clause in witness.body:
        if clause.demand not in seen:
          pending.append(clause.demand)

  return DemandRewrite(
    seeds=stable_seeds,
    demands=tuple(demands),
    rule_witnesses=tuple(rule_witnesses),
    fact_leaves=tuple(fact_leaves),
  )


def ground_one_hop_witnesses(
  witness: RuleWitness,
  rows: Iterable[TemporalIntervalRow],
) -> tuple[GroundedRuleWitness, ...]:
  '''Evaluate one selected rule against final materialized temporal rows.

  This query runs only for a demanded witness.  It performs the source-rule
  natural join and interval tests but does not recursively reconstruct child
  witnesses; callers use the resulting bindings to seed those child demands.
  '''

  ordered_rows = tuple(sorted(rows, key=_row_key))
  partial: list[dict[str, str]] = [dict(witness.head_bindings)]
  for clause in witness.body:
    next_partial: list[dict[str, str]] = []
    for bindings in partial:
      for row in ordered_rows:
        extended = _match_clause_row(clause, bindings, row)
        if extended is not None:
          next_partial.append(extended)
    partial = _deduplicate_bindings(next_partial)
    if not partial:
      break

  variable_order = _rule_variable_order(witness)
  grounded: list[GroundedRuleWitness] = []
  for bindings in partial:
    if any(variable not in bindings for variable in variable_order):
      continue
    grounded.append(
      GroundedRuleWitness(
        rule=witness,
        bindings=tuple((variable, bindings[variable]) for variable in variable_order),
      )
    )
  return tuple(grounded)


def pyreason_clause_components(
  witness: GroundedRuleWitness,
  rows: Iterable[TemporalIntervalRow],
) -> tuple[PyReasonClauseComponent, ...]:
  '''Reconstruct PyReason ``Clause-N`` cells for one grounded witness.

  Node clauses become lists of node names and binary edge clauses become lists
  of endpoint tuples, matching PyReason's dataframe cell shape.  A missing
  qualifying row means the supplied grounding is not a witness and is rejected.
  '''

  bindings = dict(witness.bindings)
  ordered_rows = tuple(sorted(rows, key=_row_key))
  components: list[PyReasonClauseComponent] = []
  for clause in witness.rule.body:
    expected = tuple(bindings[term] for term in clause.terms)
    matching = [
      row
      for row in ordered_rows
      if row.predicate == clause.predicate
      and row.arguments == expected
      and row.time == clause.time
      and _interval_satisfies(row, clause.lower, clause.upper)
    ]
    if not matching:
      raise ValueError(
        f'grounding has no qualifying row for Clause-{clause.position} '
        f'{clause.predicate}{expected} at time {clause.time}'
      )
    if len(expected) == 1:
      entities: tuple[TraceEntity, ...] = (expected[0],)
    else:
      entities = (expected,)
    components.append(
      PyReasonClauseComponent(
        position=clause.position,
        predicate=clause.predicate,
        entities=entities,
      )
    )
  return tuple(components)


def _select_rule(
  rule: SourceRule,
  rule_index: int,
  demand: DemandPattern,
) -> RuleWitness | None:
  if rule.head_predicate != demand.predicate:
    return None
  if len(rule.head_terms) != len(demand.arguments):
    return None
  if rule.delay > demand.time:
    return None

  bindings: dict[str, str] = {}
  for term, argument in zip(rule.head_terms, demand.arguments):
    if argument is None:
      continue
    previous = bindings.get(term)
    if previous is not None and previous != argument:
      return None
    bindings[term] = argument

  body_time = demand.time - rule.delay
  source_positions = rule.clause_source_positions or tuple(range(1, len(rule.clauses) + 1))
  body = tuple(
    ClauseDemand(
      position=position,
      predicate=clause.predicate,
      terms=clause.terms,
      arguments=tuple(bindings.get(term) for term in clause.terms),
      time=body_time,
      lower=clause.lower,
      upper=clause.upper,
    )
    for position, clause in zip(source_positions, rule.clauses)
  )
  if rule.head_annotation is None:
    transform: IntervalTransform = ConstantIntervalTransform(
      rule.head_lower,
      rule.head_upper,
    )
  else:
    transform = GroupedCallbackTransform(
      callback=GroupedCallbackMarker(rule.head_annotation),
      weights=rule.weights,
    )
  return RuleWitness(
    identity=RuleWitnessIdentity(rule_index, rule.name, demand),
    source_text=rule.text,
    head_predicate=rule.head_predicate,
    head_terms=rule.head_terms,
    head_bindings=tuple(
      (variable, bindings[variable])
      for variable in _first_occurrences(rule.head_terms)
      if variable in bindings
    ),
    body=body,
    delay=rule.delay,
    interval_transform=transform,
    infer_edges=rule.infer_edges,
    set_static=rule.set_static,
  )


def _matching_fact_leaves(
  source: SourceProgram,
  demand: DemandPattern,
) -> list[FactLeaf]:
  leaves: list[FactLeaf] = []
  for fact_index, fact in enumerate(source.facts):
    if fact.end_time < fact.start_time:
      continue
    if fact.predicate != demand.predicate:
      continue
    if not _arguments_match(demand.arguments, fact.arguments):
      continue
    if demand.time < fact.start_time:
      continue
    if not fact.static and demand.time > fact.end_time:
      continue
    leaves.append(
      FactLeaf(
        identity=FactWitnessIdentity(
          fact_index=fact_index,
          fact_name=fact.name,
          arguments=fact.arguments,
          time=demand.time,
        ),
        source_text=fact.text,
        predicate=fact.predicate,
        lower=fact.lower,
        upper=fact.upper,
        static=fact.static,
        demanded_by=demand,
      )
    )
  return leaves


def _arguments_match(
  pattern: tuple[str | None, ...],
  arguments: tuple[str, ...],
) -> bool:
  return len(pattern) == len(arguments) and all(
    expected is None or expected == actual for expected, actual in zip(pattern, arguments)
  )


def _match_clause_row(
  clause: ClauseDemand,
  bindings: Mapping[str, str],
  row: TemporalIntervalRow,
) -> dict[str, str] | None:
  if row.predicate != clause.predicate or row.time != clause.time:
    return None
  if len(row.arguments) != len(clause.terms):
    return None
  if not _interval_satisfies(row, clause.lower, clause.upper):
    return None
  extended = dict(bindings)
  for term, argument in zip(clause.terms, row.arguments):
    previous = extended.get(term)
    if previous is not None and previous != argument:
      return None
    extended[term] = argument
  return extended


def _interval_satisfies(
  row: TemporalIntervalRow,
  required_lower: float,
  required_upper: float,
) -> bool:
  # PyReason asks whether the materialized world interval is contained in the
  # clause interval, i.e. ``world_bound in clause_bound``.
  return bool(required_lower <= row.lower and row.upper <= required_upper)


def _deduplicate_bindings(bindings: Iterable[dict[str, str]]) -> list[dict[str, str]]:
  result: list[dict[str, str]] = []
  seen: set[tuple[tuple[str, str], ...]] = set()
  for binding in bindings:
    key = tuple(sorted(binding.items()))
    if key in seen:
      continue
    seen.add(key)
    result.append(binding)
  return result


def _rule_variable_order(witness: RuleWitness) -> tuple[str, ...]:
  return _first_occurrences(
    (*witness.head_terms, *(term for clause in witness.body for term in clause.terms))
  )


def _first_occurrences(variables: tuple[str, ...]) -> tuple[str, ...]:
  return tuple(dict.fromkeys(variables))


def _row_key(row: TemporalIntervalRow) -> tuple[object, ...]:
  return (
    row.predicate,
    row.time,
    row.arguments,
    row.lower,
    row.upper,
  )
