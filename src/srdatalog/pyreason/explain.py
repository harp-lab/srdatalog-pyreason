'''On-demand recursive explanations over a materialized PyReason result.

Ordinary evaluation stores no provenance payload.  A caller supplies grounded
atoms to explain; the demand rewrite then selects and grounds only reachable
rules and recursively propagates grounded body demands until source facts.
'''

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import cast

from .demand import (
  ConstantIntervalTransform,
  DemandSeed,
  GroundedRuleWitness,
  GroupedCallbackTransform,
  PyReasonClauseComponent,
  RuleWitness,
  TraceEntity,
  ground_one_hop_witnesses,
  pyreason_clause_components,
  rewrite_for_demands,
)
from .model import RuleCandidateRow, SourceProgram, TemporalIntervalRow


@dataclass(frozen=True)
class FactExplanation:
  fact_index: int
  fact_name: str
  lower: float
  upper: float


@dataclass(frozen=True)
class IPLExplanation:
  '''One logical inconsistent-predicate-list derivation.'''

  pair_index: int
  direction: int
  source: DemandSeed
  candidate_lower: float
  candidate_upper: float

  @property
  def witness_key(self) -> tuple[object, ...]:
    return (
      "ipl",
      self.pair_index,
      self.direction,
      self.source,
      self.candidate_lower,
      self.candidate_upper,
    )


@dataclass(frozen=True)
class ClosedWorldAssumptionExplanation:
  '''A transient default-false read, not a stored source tuple.'''

  atom: DemandSeed
  source: DemandSeed | None = None

  @property
  def witness_key(self) -> tuple[object, ...]:
    base: tuple[object, ...] = (
      "closed-world",
      self.atom.predicate,
      self.atom.arguments,
      self.atom.time,
    )
    if not self.atom.has_observed_interval:
      return base
    versioned = (
      *base,
      self.atom.observed_lower,
      self.atom.observed_upper,
      self.atom.evidence_mode,
    )
    return versioned if self.source is None else (*versioned, self.source)


@dataclass(frozen=True)
class RuleExplanation:
  head: DemandSeed
  rule_index: int
  rule_name: str
  bindings: tuple[tuple[str, str], ...]
  groundings: tuple[tuple[tuple[str, str], ...], ...]
  candidate_lower: float
  candidate_upper: float
  clauses: tuple[PyReasonClauseComponent, ...]
  children: tuple[DemandSeed, ...]
  rank: int | None = None
  grounding_body_intervals: tuple[tuple[tuple[float, float], ...], ...] = ()
  closed_world_clause_positions: tuple[int, ...] = ()

  @property
  def witness_key(self) -> tuple[object, ...]:
    '''Stable logical witness identity; never a physical GPU row ID.'''

    base = (
      self.rule_index,
      self.head.predicate,
      self.head.arguments,
      self.head.time,
      self.bindings,
      self.groundings,
      self.candidate_lower,
      self.candidate_upper,
      tuple((clause.position, clause.predicate, clause.entities) for clause in self.clauses),
    )
    if (
      self.rank is None
      and not self.grounding_body_intervals
      and not self.closed_world_clause_positions
    ):
      return base
    return (
      *base,
      self.rank,
      self.grounding_body_intervals,
      self.closed_world_clause_positions,
    )


@dataclass(frozen=True)
class TemporalTransitionExplanation:
  '''One compiler-generated world transition used by on-demand provenance.'''

  kind: str
  source: DemandSeed

  @property
  def witness_key(self) -> tuple[object, ...]:
    return ("temporal-transition", self.kind, self.source)


@dataclass(frozen=True)
class AtomExplanation:
  demand: DemandSeed
  materialized_lower: float
  materialized_upper: float
  inconsistent: bool
  facts: tuple[FactExplanation, ...]
  rules: tuple[RuleExplanation, ...]
  ipls: tuple[IPLExplanation, ...] = ()
  transitions: tuple[TemporalTransitionExplanation, ...] = ()
  closed_world_assumptions: tuple[ClosedWorldAssumptionExplanation, ...] = ()


@dataclass(frozen=True)
class ExplanationGraph:
  roots: tuple[DemandSeed, ...]
  atoms: tuple[AtomExplanation, ...]

  def atom(self, demand: DemandSeed) -> AtomExplanation | None:
    exact = next((atom for atom in self.atoms if atom.demand == demand), None)
    if exact is not None or demand.has_observed_interval:
      return exact
    # Preserve the convenient atom/time lookup for callers that do not care
    # which historical occurrence they inspect.  Use ``occurrences`` when a
    # proof contains more than one interval version of the same atom.
    return next(
      (
        atom
        for atom in self.atoms
        if atom.demand.predicate == demand.predicate
        and atom.demand.arguments == demand.arguments
        and atom.demand.time == demand.time
      ),
      None,
    )

  def occurrences(self, demand: DemandSeed) -> tuple[AtomExplanation, ...]:
    '''Return every logical interval occurrence for one grounded atom/time.'''
    return tuple(
      atom
      for atom in self.atoms
      if atom.demand.predicate == demand.predicate
      and atom.demand.arguments == demand.arguments
      and atom.demand.time == demand.time
    )


def explain_demands(
  source: SourceProgram,
  materialized_rows: Iterable[TemporalIntervalRow],
  seeds: Iterable[DemandSeed],
  *,
  rule_candidates: Iterable[RuleCandidateRow] | None = None,
) -> ExplanationGraph:
  '''Recursively explain grounded seeds without changing base materialization.

  When ``rule_candidates`` is supplied, rule explanations come from an
  instrumented on-demand replay.  Their body endpoint columns retain transient
  interval versions that cannot be reconstructed from the final world state.
  ``None`` keeps the legacy final-snapshot path for callers that only need a
  static explanation.
  '''

  roots = tuple(dict.fromkeys(seeds))
  materialized = tuple(materialized_rows)
  retained_candidates = None if rule_candidates is None else tuple(rule_candidates)
  read_rows = _effective_rows(source, materialized, roots)
  pending: deque[DemandSeed] = deque(roots)
  seen: set[DemandSeed] = set()
  atoms: list[AtomExplanation] = []

  while pending:
    demand = pending.popleft()
    if demand in seen:
      continue
    seen.add(demand)
    if demand.evidence_mode == 'closed-world-default':
      materialized_source = next(
        (
          row
          for row in materialized
          if row.predicate == demand.predicate
          and row.arguments == demand.arguments
          and row.time == demand.time
          and ((row.lower, row.upper) == (0.0, 1.0) or row.frozen or row.inconsistent)
        ),
        None,
      )
      source_demand = None
      if materialized_source is not None:
        observed_lower = (
          materialized_source.raw_lower
          if materialized_source.raw_lower is not None
          else materialized_source.lower
        )
        observed_upper = (
          materialized_source.raw_upper
          if materialized_source.raw_upper is not None
          else materialized_source.upper
        )
        source_demand = DemandSeed(
          demand.predicate,
          demand.arguments,
          demand.time,
          observed_lower=observed_lower,
          observed_upper=observed_upper,
          evidence_mode='materialized',
        )
        if source_demand not in seen:
          pending.append(source_demand)
      atoms.append(
        AtomExplanation(
          demand=demand,
          # Preserve the stored range occurrence.  The leaf below records why
          # PyReason interpreted this otherwise-unknown range as effective
          # false for the closed-world clause.
          materialized_lower=0.0,
          materialized_upper=1.0,
          inconsistent=False,
          facts=(),
          rules=(),
          closed_world_assumptions=(ClosedWorldAssumptionExplanation(demand, source_demand),),
        )
      )
      continue
    matching_rows = tuple(
      row
      for row in materialized
      if row.predicate == demand.predicate
      and row.arguments == demand.arguments
      and row.time == demand.time
    )
    visible = _visible_occurrence(demand, matching_rows)
    if visible is None:
      continue
    rewrite = rewrite_for_demands(source, (demand,))
    exact_pattern = (demand.predicate, demand.arguments, demand.time)
    direct = tuple(
      witness
      for witness in rewrite.rule_witnesses
      if (
        witness.identity.demanded_head.predicate,
        witness.identity.demanded_head.arguments,
        witness.identity.demanded_head.time,
      )
      == exact_pattern
    )
    rule_explanations: list[RuleExplanation] = []
    for witness in direct:
      if retained_candidates is not None:
        matching_candidates = tuple(
          candidate
          for candidate in retained_candidates
          if candidate.rule_index == witness.identity.rule_index
          and candidate.head_predicate == demand.predicate
          and candidate.head_arguments == demand.arguments
          and candidate.head_time == demand.time
          and _candidate_supports_occurrence(candidate, demand)
        )
        candidate_explanations = tuple(
          _candidate_rule_explanation(witness, event, demand)
          for event in _candidate_events(witness, matching_candidates)
        )
        rule_explanations.extend(candidate_explanations)
        for explanation in candidate_explanations:
          for child in explanation.children:
            if child not in seen:
              pending.append(child)
        continue
      grounded = ground_one_hop_witnesses(witness, read_rows)
      if not grounded:
        continue
      explanation = _rule_explanation(
        source,
        grounded,
        read_rows,
        visible,
        materialized,
      )
      rule_explanations.append(explanation)
      for child in explanation.children:
        if child not in seen:
          pending.append(child)
    facts = tuple(
      FactExplanation(
        fact_index=leaf.identity.fact_index,
        fact_name=leaf.identity.fact_name,
        lower=leaf.lower,
        upper=leaf.upper,
      )
      for leaf in rewrite.fact_leaves
      if leaf.demanded_by.predicate == demand.predicate
      and leaf.demanded_by.arguments == demand.arguments
      and leaf.demanded_by.time == demand.time
      and _interval_supports_occurrence(leaf.lower, leaf.upper, demand)
    )
    ipls = _ipl_explanations(source, materialized, demand)
    for ipl in ipls:
      if ipl.source not in seen:
        pending.append(ipl.source)
    transitions = _temporal_transitions(source, materialized, demand, visible)
    for transition in transitions:
      if transition.source not in seen:
        pending.append(transition.source)
    closed_world_assumptions = (
      (ClosedWorldAssumptionExplanation(demand),)
      if _is_closed_world_default(source, demand, visible)
      else ()
    )
    atoms.append(
      AtomExplanation(
        demand=demand,
        materialized_lower=visible.lower,
        materialized_upper=visible.upper,
        inconsistent=(
          visible.inconsistent
          if demand.has_observed_interval
          else any(row.inconsistent for row in matching_rows)
        ),
        facts=facts,
        rules=tuple(rule_explanations),
        ipls=ipls,
        transitions=transitions,
        closed_world_assumptions=closed_world_assumptions,
      )
    )
  return ExplanationGraph(roots=roots, atoms=tuple(atoms))


def _candidate_rule_explanation(
  witness: RuleWitness,
  candidates: tuple[RuleCandidateRow, ...],
  head: DemandSeed,
) -> RuleExplanation:
  '''Decode one retained aggregate winner and all of its member evidence.'''
  candidate = candidates[0]
  clause_entities: dict[tuple[int, str], list[TraceEntity]] = {}
  children: list[DemandSeed] = []
  for grounding in candidates:
    bindings = dict(grounding.bindings)
    for clause_index, clause in enumerate(witness.body):
      arguments = tuple(bindings[term] for term in clause.terms)
      entity: TraceEntity = arguments[0] if len(arguments) == 1 else arguments
      bucket = clause_entities.setdefault((clause.position, clause.predicate), [])
      if entity not in bucket:
        bucket.append(entity)
      observed_lower, observed_upper = grounding.body_intervals[clause_index]
      closed_world_default = clause_index in grounding.closed_world_clause_positions and (
        (observed_lower, observed_upper) == (0.0, 1.0) or observed_lower > observed_upper
      )
      children.append(
        DemandSeed(
          predicate=clause.predicate,
          arguments=arguments,
          time=clause.time,
          observed_lower=observed_lower,
          observed_upper=observed_upper,
          evidence_mode=('closed-world-default' if closed_world_default else 'materialized'),
        )
      )
  clauses = tuple(
    PyReasonClauseComponent(position, predicate, tuple(entities))
    for (position, predicate), entities in sorted(clause_entities.items())
  )
  return RuleExplanation(
    head=head,
    rule_index=candidate.rule_index,
    rule_name=candidate.rule_name,
    bindings=witness.head_bindings,
    groundings=tuple(item.bindings for item in candidates),
    candidate_lower=candidate.candidate_lower,
    candidate_upper=candidate.candidate_upper,
    clauses=clauses,
    children=tuple(dict.fromkeys(children)),
    rank=candidate.rank,
    grounding_body_intervals=tuple(item.body_intervals for item in candidates),
    closed_world_clause_positions=candidate.closed_world_clause_positions,
  )


def _candidate_events(
  witness: RuleWitness,
  candidates: tuple[RuleCandidateRow, ...],
) -> tuple[tuple[RuleCandidateRow, ...], ...]:
  '''Group member rows beneath one selected aggregate occurrence.'''
  if isinstance(witness.interval_transform, ConstantIntervalTransform):
    # Ordinary Datalog witnesses remain distinct by grounding and observed
    # interval version.  Grouping is a source callback operation, not a
    # blanket quotient over every rule that happens to share a head tuple.
    return tuple(
      (candidate,) for candidate in sorted(candidates, key=lambda item: item.witness_key)
    )
  grouped: dict[tuple[object, ...], list[RuleCandidateRow]] = {}
  for candidate in candidates:
    key = (
      candidate.rule_index,
      candidate.head_predicate,
      candidate.head_arguments,
      candidate.head_time,
      candidate.rank,
      candidate.candidate_lower,
      candidate.candidate_upper,
    )
    grouped.setdefault(key, []).append(candidate)
  return tuple(
    tuple(sorted(members, key=lambda item: item.witness_key))
    for _, members in sorted(grouped.items(), key=lambda item: item[0])
  )


def _candidate_supports_occurrence(
  candidate: RuleCandidateRow,
  demand: DemandSeed,
) -> bool:
  return _interval_supports_occurrence(
    candidate.candidate_lower,
    candidate.candidate_upper,
    demand,
  )


def _interval_supports_occurrence(
  lower: float,
  upper: float,
  demand: DemandSeed,
) -> bool:
  if not demand.has_observed_interval:
    return True
  assert demand.observed_lower is not None
  assert demand.observed_upper is not None
  if demand.observed_lower > demand.observed_upper:
    # A crossed lattice value is the join of conflicting contributors; retain
    # all demanded candidates for the key rather than inventing row identity.
    return True
  return lower <= demand.observed_lower and demand.observed_upper <= upper


def _visible_occurrence(
  demand: DemandSeed,
  matching_rows: tuple[TemporalIntervalRow, ...],
) -> TemporalIntervalRow | None:
  if not demand.has_observed_interval:
    return matching_rows[0] if matching_rows else None
  assert demand.observed_lower is not None
  assert demand.observed_upper is not None
  matching = next(
    (
      row
      for row in matching_rows
      if (row.raw_lower if row.raw_lower is not None else row.lower) == demand.observed_lower
      and (row.raw_upper if row.raw_upper is not None else row.upper) == demand.observed_upper
    ),
    None,
  )
  if matching is not None:
    return matching
  inconsistent = demand.observed_lower > demand.observed_upper
  return TemporalIntervalRow(
    predicate=demand.predicate,
    arguments=demand.arguments,
    time=demand.time,
    lower=0.0 if inconsistent else demand.observed_lower,
    upper=1.0 if inconsistent else demand.observed_upper,
    inconsistent=inconsistent,
    frozen=inconsistent,
    raw_lower=demand.observed_lower if inconsistent else None,
    raw_upper=demand.observed_upper if inconsistent else None,
  )


def _temporal_transitions(
  source: SourceProgram,
  rows: tuple[TemporalIntervalRow, ...],
  demand: DemandSeed,
  visible: TemporalIntervalRow,
) -> tuple[TemporalTransitionExplanation, ...]:
  '''Reconstruct frame/reset/freeze edges only for a demanded occurrence.'''

  if demand.time == 0:
    return ()
  previous = next(
    (
      row
      for row in rows
      if row.predicate == demand.predicate
      and row.arguments == demand.arguments
      and row.time == demand.time - 1
    ),
    None,
  )
  if previous is None:
    return ()
  previous_lower = previous.raw_lower if previous.raw_lower is not None else previous.lower
  previous_upper = previous.raw_upper if previous.raw_upper is not None else previous.upper
  source_demand = DemandSeed(
    demand.predicate,
    demand.arguments,
    demand.time - 1,
    observed_lower=previous_lower,
    observed_upper=previous_upper,
    evidence_mode='materialized',
  )
  if visible.frozen and (previous.frozen or previous.inconsistent):
    return (TemporalTransitionExplanation('conflict-freeze', source_demand),)
  if dict(source.settings).get('persistent', False):
    return (TemporalTransitionExplanation('persistent-frame', source_demand),)
  frozen_by_static_fact = any(
    fact.static
    and fact.end_time >= fact.start_time
    and fact.predicate == demand.predicate
    and fact.arguments == demand.arguments
    and fact.start_time <= demand.time - 1
    for fact in source.facts
  )
  if (
    not frozen_by_static_fact
    and not previous.frozen
    and (visible.lower, visible.upper) == (0.0, 1.0)
  ):
    return (TemporalTransitionExplanation('label-reset', source_demand),)
  return ()


def _ipl_explanations(
  source: SourceProgram,
  rows: tuple[TemporalIntervalRow, ...],
  demand: DemandSeed,
) -> tuple[IPLExplanation, ...]:
  result: list[IPLExplanation] = []
  by_key = {(row.predicate, row.arguments, row.time): row for row in rows}
  for pair_index, (left, right) in enumerate(source.inconsistent_predicates):
    directions = []
    if demand.predicate == right:
      directions.append((0, left))
    if demand.predicate == left:
      directions.append((1, right))
    for direction, source_predicate in directions:
      source_row = by_key.get((source_predicate, demand.arguments, demand.time))
      if source_row is None:
        continue
      if source_row.inconsistent:
        candidate_lower = (
          source_row.raw_lower if source_row.raw_lower is not None else source_row.lower
        )
        candidate_upper = (
          source_row.raw_upper if source_row.raw_upper is not None else source_row.upper
        )
      elif (source_row.lower, source_row.upper) == (0.0, 1.0):
        # Lattice bottom and CWA range rows are not IPL updates in PyReason.
        continue
      else:
        candidate_lower = 1.0 - source_row.upper
        candidate_upper = 1.0 - source_row.lower
      result.append(
        IPLExplanation(
          pair_index=pair_index,
          direction=direction,
          source=DemandSeed(source_predicate, demand.arguments, demand.time),
          candidate_lower=candidate_lower,
          candidate_upper=candidate_upper,
        )
      )
  return tuple(result)


def _is_closed_world_default(
  source: SourceProgram,
  demand: DemandSeed,
  visible: TemporalIntervalRow,
) -> bool:
  if demand.predicate not in source.closed_world_predicates:
    return False
  if (visible.lower, visible.upper) != (0.0, 0.0):
    return False
  for fact in source.facts:
    applies = (
      fact.end_time >= fact.start_time
      and fact.start_time <= demand.time
      and (fact.static or demand.time <= fact.end_time)
    )
    if (
      applies
      and fact.predicate == demand.predicate
      and fact.arguments == demand.arguments
      and (fact.lower, fact.upper) != (0.0, 1.0)
    ):
      return False
  return True


def _rule_explanation(
  source: SourceProgram,
  grounded_witnesses: tuple[GroundedRuleWitness, ...],
  rows: tuple[TemporalIntervalRow, ...],
  materialized_head: TemporalIntervalRow,
  materialized_rows: tuple[TemporalIntervalRow, ...],
) -> RuleExplanation:
  witness = grounded_witnesses[0]
  transform = witness.rule.interval_transform
  if isinstance(transform, ConstantIntervalTransform):
    candidate_lower = transform.lower
    candidate_upper = transform.upper
  elif isinstance(transform, GroupedCallbackTransform):
    candidate_lower, candidate_upper = _evaluate_grouped_callback(
      source,
      transform,
      grounded_witnesses,
      rows,
    )
  else:
    raise TypeError(f"unsupported interval transform {type(transform).__name__}")
  all_bindings = [dict(grounded.bindings) for grounded in grounded_witnesses]
  children_list: list[DemandSeed] = []
  for bindings in all_bindings:
    for clause in witness.rule.body:
      arguments = tuple(bindings[term] for term in clause.terms)
      actual = next(
        (
          row
          for row in materialized_rows
          if row.predicate == clause.predicate
          and row.arguments == arguments
          and row.time == clause.time
        ),
        None,
      )
      closed_world_default = (
        clause.predicate in source.closed_world_predicates
        and clause.lower == 0.0
        and (
          actual is None
          or actual.frozen
          or actual.inconsistent
          or (actual.lower, actual.upper) == (0.0, 1.0)
        )
      )
      if closed_world_default:
        observed_lower = (
          actual.raw_lower if actual is not None and actual.raw_lower is not None else 0.0
        )
        observed_upper = (
          actual.raw_upper if actual is not None and actual.raw_upper is not None else 1.0
        )
        children_list.append(
          DemandSeed(
            predicate=clause.predicate,
            arguments=arguments,
            time=clause.time,
            observed_lower=observed_lower,
            observed_upper=observed_upper,
            evidence_mode='closed-world-default',
          )
        )
      else:
        children_list.append(
          DemandSeed(
            predicate=clause.predicate,
            arguments=arguments,
            time=clause.time,
          )
        )
  children = tuple(dict.fromkeys(children_list))
  clause_sets: dict[tuple[int, str], list[TraceEntity]] = {}
  for grounded in grounded_witnesses:
    for component in pyreason_clause_components(grounded, rows):
      bucket = clause_sets.setdefault((component.position, component.predicate), [])
      for entity in component.entities:
        if entity not in bucket:
          bucket.append(entity)
  clauses = tuple(
    PyReasonClauseComponent(position, predicate, tuple(entities))
    for (position, predicate), entities in sorted(clause_sets.items())
  )
  return RuleExplanation(
    head=DemandSeed(
      materialized_head.predicate,
      materialized_head.arguments,
      materialized_head.time,
    ),
    rule_index=witness.rule.identity.rule_index,
    rule_name=witness.rule.identity.rule_name,
    bindings=witness.rule.head_bindings,
    groundings=tuple(grounded.bindings for grounded in grounded_witnesses),
    candidate_lower=candidate_lower,
    candidate_upper=candidate_upper,
    clauses=clauses,
    children=children,
  )


@dataclass(frozen=True)
class _CallbackInterval:
  lower: float
  upper: float


@dataclass(frozen=True)
class _CallbackLabel:
  value: str


def _evaluate_grouped_callback(
  source: SourceProgram,
  transform: GroupedCallbackTransform,
  grounded_witnesses: tuple[GroundedRuleWitness, ...],
  rows: tuple[TemporalIntervalRow, ...],
) -> tuple[float, float]:
  callback = dict(source.annotation_functions).get(transform.callback.annotation_name)
  if callback is None:
    raise ValueError(
      f"missing annotation callback {transform.callback.annotation_name!r} during replay"
    )
  python_callback = cast(Callable[..., object], getattr(callback, "py_func", callback))
  rule = grounded_witnesses[0].rule
  annotations: list[list[_CallbackInterval]] = []
  qualified_nodes: list[list[str]] = []
  qualified_edges: list[list[tuple[str, ...]]] = []
  for clause in rule.body:
    clause_annotations: list[_CallbackInterval] = []
    nodes: list[str] = []
    edges: list[tuple[str, ...]] = []
    seen_components: set[tuple[str, ...]] = set()
    for grounded in grounded_witnesses:
      bindings = dict(grounded.bindings)
      arguments = tuple(bindings[term] for term in clause.terms)
      if arguments in seen_components:
        continue
      row = next(
        (
          item
          for item in rows
          if item.predicate == clause.predicate
          and item.arguments == arguments
          and item.time == clause.time
          and clause.lower <= item.lower
          and item.upper <= clause.upper
        ),
        None,
      )
      if row is None:
        continue
      seen_components.add(arguments)
      clause_annotations.append(_CallbackInterval(row.lower, row.upper))
      if len(arguments) == 1:
        nodes.append(arguments[0])
      else:
        edges.append(arguments)
    annotations.append(clause_annotations)
    qualified_nodes.append(nodes)
    qualified_edges.append(edges)
  result = python_callback(
    annotations,
    list(transform.weights),
    qualified_nodes,
    qualified_edges,
    [_CallbackLabel(clause.predicate) for clause in rule.body],
    [clause.terms for clause in rule.body],
  )
  if not isinstance(result, (tuple, list)) or len(result) != 2:
    raise ValueError(f"annotation {transform.callback.annotation_name!r} did not return two bounds")
  return min(max(float(result[0]), 0.0), 1.0), min(max(float(result[1]), 0.0), 1.0)


def _effective_rows(
  source: SourceProgram,
  rows: tuple[TemporalIntervalRow, ...],
  roots: tuple[DemandSeed, ...],
) -> tuple[TemporalIntervalRow, ...]:
  '''Build PyReason's transient CWA read view for explanation joins.

  Materialized rows are retained verbatim.  A separate effective-false row is
  added only for clause evaluation when the label is missing, explicitly
  unknown, or conflict-frozen.  Atom explanations therefore never confuse the
  read-time CWA coercion with stored state.
  '''

  if not source.closed_world_predicates:
    return rows
  arities = _predicate_arities(source)
  max_time = max(
    (row.time for row in rows),
    default=max((root.time for root in roots), default=0),
  )
  retained = list(rows)
  rows_by_key: dict[tuple[str, tuple[str, ...], int], list[TemporalIntervalRow]] = {}
  for row in rows:
    if row.predicate in source.closed_world_predicates:
      rows_by_key.setdefault((row.predicate, row.arguments, row.time), []).append(row)
  node_births = dict(source.node_birth_times)
  edge_births = {
    (source_node, target_node): birth for source_node, target_node, birth in source.edge_birth_times
  }
  for predicate in sorted(source.closed_world_predicates):
    arity = arities.get(predicate)
    if arity == 1:

      def domain_at(time: int) -> tuple[tuple[str, ...], ...]:
        return tuple((node,) for node in source.node_domain if node_births.get(node, 0) <= time)
    elif arity == 2:

      def domain_at(time: int) -> tuple[tuple[str, ...], ...]:
        return tuple(edge for edge in source.edge_domain if edge_births.get(edge, 0) <= time)
    else:
      continue
    for time in range(max_time + 1):
      for arguments in domain_at(time):
        key = (predicate, arguments, time)
        actual = rows_by_key.get(key, ())
        if actual and not any(
          row.frozen or row.inconsistent or (row.lower, row.upper) == (0.0, 1.0) for row in actual
        ):
          continue
        retained.append(
          TemporalIntervalRow(
            predicate=predicate,
            arguments=arguments,
            time=time,
            lower=0.0,
            upper=0.0,
          )
        )
  return tuple(sorted(retained, key=_row_key))


def _predicate_arities(source: SourceProgram) -> dict[str, int]:
  result: dict[str, int] = {}
  for fact in source.facts:
    result.setdefault(fact.predicate, len(fact.arguments))
  for rule in source.rules:
    result.setdefault(rule.head_predicate, len(rule.head_terms))
    for clause in rule.clauses:
      result.setdefault(clause.predicate, len(clause.terms))
  return result


def _row_key(row: TemporalIntervalRow) -> tuple[object, ...]:
  return (row.time, row.predicate, row.arguments, row.lower, row.upper)
