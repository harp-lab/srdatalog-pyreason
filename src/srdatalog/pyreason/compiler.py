"""Compile captured PyReason source directly to native SRDatalog Core."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .demand import DemandSeed, rewrite_for_demands
from .model import (
  NativePlan,
  RuleWitnessOutput,
  SourceProgram,
  TemporalIntervalOutput,
)
from .rule_compiler import (
  RuleCompileError,
  WitnessRelations,
  callback_source,
  compile_rule,
)


class UnsupportedAnnotationSemantics(NotImplementedError):
  """The source program uses annotation behavior outside the compiler IR."""


def _effective_source(source: SourceProgram) -> SourceProgram:
  """Canonicalize the rule order that PyReason 3.6 actually evaluates.

  PyReason stably moves unary clauses before binary clauses when its loaded
  graph has more edges than nodes.  Extended callbacks observe that reordered
  annotation/clause array while the weights stay in source order.  Preserve
  the original one-based positions separately for demand-driven presentation.
  """

  reorder = source.reorder_clauses_node_first
  if reorder is None:
    reorder = len(source.edge_domain) > len(source.node_domain)
  if not reorder:
    return source
  effective_rules = []
  changed = False
  for rule in source.rules:
    positions = rule.clause_source_positions or tuple(range(1, len(rule.clauses) + 1))
    ordered = sorted(
      zip(rule.clauses, positions),
      key=lambda item: 0 if len(item[0].terms) == 1 else 1,
    )
    clauses = tuple(clause for clause, _ in ordered)
    source_positions = tuple(position for _, position in ordered)
    effective_rules.append(
      replace(
        rule,
        clauses=clauses,
        clause_source_positions=source_positions,
      )
    )
    changed = changed or clauses != rule.clauses or not rule.clause_source_positions
  return replace(source, rules=tuple(effective_rules)) if changed else source


def _fact_has_occurrence(fact: Any, timesteps: int) -> bool:
  """Whether PyReason's initial ``range(start, end + 1)`` schedules the fact."""

  return bool(0 <= fact.start_time <= timesteps and fact.end_time >= fact.start_time)


def compile_source(
  source: SourceProgram,
  *,
  timesteps: int,
  output_root: str | Path,
  provenance_demands: Iterable[DemandSeed] = (),
) -> NativePlan:
  """Compile one neutral source program without application-name dispatch."""
  stable_demands = tuple(dict.fromkeys(provenance_demands))
  if timesteps < 0:
    raise UnsupportedAnnotationSemantics(
      "native PyReason compilation requires a finite timestep horizon"
    )
  source = _effective_source(source)
  settings = dict(source.settings)
  negative_fact_times = tuple(
    fact.name for fact in source.facts if fact.start_time < 0 or fact.end_time < 0
  )
  if negative_fact_times:
    raise UnsupportedAnnotationSemantics(
      "negative fact times are cast to uint16 by PyReason 3.6 and are not a "
      f"stable temporal semantics ({', '.join(map(repr, negative_fact_times))})"
    )
  if settings.get("inconsistency_check", True) is False:
    raise UnsupportedAnnotationSemantics(
      "inconsistency_check=False uses PyReason's order-dependent conflicting "
      "update behavior and is outside the monotone interval lattice"
    )
  if settings.get("abort_on_inconsistency", False):
    raise UnsupportedAnnotationSemantics(
      "abort_on_inconsistency=True requires update-time short-circuit semantics; "
      "the native monotone materializer does not silently approximate it"
    )
  node_symbols = set(source.node_domain)
  ground_sensitive_terms = tuple(
    sorted(
      {
        term
        for rule in source.rules
        for term in (
          *rule.head_terms,
          *(term for clause in rule.clauses for term in clause.terms),
        )
        if term in node_symbols
      }
    )
  )
  if settings.get("allow_ground_rules", False) and ground_sensitive_terms:
    raise UnsupportedAnnotationSemantics(
      "allow_ground_rules=True treats rule terms matching graph components as "
      "constants; the captured rule does not yet mark those term occurrences "
      f"({', '.join(map(repr, ground_sensitive_terms))})"
    )
  if source.inconsistent_predicates:
    raise UnsupportedAnnotationSemantics(
      "inconsistent-predicate-list updates require atomic, one-hop co-commit "
      "with the primary update; the previous recursive IPL approximation has "
      "been disabled rather than returning phase-shifted facts"
    )
  inferred_edges = tuple(
    rule.name for rule in source.rules if rule.infer_edges and len(rule.head_terms) == 2
  )
  if inferred_edges:
    names = ", ".join(repr(name) for name in inferred_edges)
    raise UnsupportedAnnotationSemantics(
      "binary infer_edges rules mutate PyReason's graph and every later active "
      f"edge domain ({names}); fixed relational heads are not an equivalent lowering"
    )
  repeated_binary_terms = tuple(
    (rule.name, "head" if repeated_head else f"body clause {clause_index + 1}")
    for rule in source.rules
    for repeated_head, clause_index in (
      ((len(rule.head_terms) == 2 and rule.head_terms[0] == rule.head_terms[1]), -1),
      *(
        (False, index)
        for index, clause in enumerate(rule.clauses)
        if len(clause.terms) == 2 and clause.terms[0] == clause.terms[1]
      ),
    )
    if repeated_head or clause_index >= 0
  )
  if repeated_binary_terms:
    locations = ", ".join(f"{name!r} {location}" for name, location in repeated_binary_terms)
    raise UnsupportedAnnotationSemantics(
      "PyReason 3.6 treats repeated variables in binary atoms as two operational "
      f"slots instead of Datalog equality ({locations}); this oracle quirk is "
      "rejected rather than compiled with different semantics"
    )

  static_predicates = {
    fact.predicate for fact in source.facts if fact.static and _fact_has_occurrence(fact, timesteps)
  }
  static_rule_heads = static_predicates & {rule.head_predicate for rule in source.rules}
  if static_rule_heads:
    names = ", ".join(sorted(static_rule_heads))
    raise UnsupportedAnnotationSemantics(
      "PyReason skips every rule update to a frozen static key; exact native "
      f"support for rule-head predicates {names} needs a static-key write guard"
    )
  overlapping_static_facts = _overlapping_static_fact_keys(source, timesteps)
  if overlapping_static_facts:
    names = ", ".join(
      f"{predicate}{arguments}" for predicate, arguments in overlapping_static_facts
    )
    raise UnsupportedAnnotationSemantics(
      "same-key facts overlap a static freeze and are source-order-sensitive in "
      f"PyReason ({names}); interval-lattice seed merging would change the result"
    )

  default_reads = _closed_world_default_reads(source)
  map_sensitive_reads = _map_sensitive_closed_world_reads(source)
  initially_labeled = {
    predicate
    for predicate in map_sensitive_reads
    if any(
      fact.predicate == predicate and fact.start_time == 0 and _fact_has_occurrence(fact, timesteps)
      for fact in source.facts
    )
  }
  dynamically_introduced = {rule.head_predicate for rule in source.rules} & map_sensitive_reads
  arities = _predicate_arities(source)
  unsafe_first_labels = {
    predicate
    for predicate in dynamically_introduced - initially_labeled
    if (
      (arities[predicate] == 1 and len(source.node_domain) > 1)
      or (arities[predicate] == 2 and len(source.edge_domain) > 1)
    )
  }
  if unsafe_first_labels:
    names = ", ".join(sorted(unsafe_first_labels))
    raise UnsupportedAnnotationSemantics(
      "unbound closed-world grounding changes from the full active domain to "
      f"the predicate map when {names} is first derived; exact support needs "
      "a source-round grounding-domain relation"
    )
  if settings.get("persistent", False) and timesteps > 0 and default_reads:
    raise UnsupportedAnnotationSemantics(
      "persistent carry must initialize each target timestep before an unbound "
      "closed-world read; the current temporal relation is not phase-stratified"
    )
  if settings.get("persistent", False) and any(rule.delay > 0 for rule in source.rules):
    raise UnsupportedAnnotationSemantics(
      "persistent delayed updates need a strict-change event after frame carry; "
      "a delayed candidate occurrence alone is only a horizon event"
    )
  persistent_temporal_facts = tuple(
    fact.name
    for fact in source.facts
    if settings.get("persistent", False)
    and not fact.static
    and fact.end_time >= fact.start_time
    and (fact.start_time > 0 or fact.end_time > 0)
  )
  if persistent_temporal_facts:
    raise UnsupportedAnnotationSemantics(
      "persistent future/repeated fact writes need a strict-change event after "
      "frame carry; a scheduled informative fact can be a no-op "
      f"({', '.join(map(repr, persistent_temporal_facts))})"
    )
  delayed_default_targets = {
    rule.head_predicate for rule in source.rules if rule.delay > 0
  } & default_reads
  if delayed_default_targets:
    names = ", ".join(sorted(delayed_default_targets))
    raise UnsupportedAnnotationSemantics(
      "delayed updates must initialize their target timestep before closed-world "
      f"grounding for {names}; exact support needs time-stratified worlds"
    )

  bottom_admitting_heads = {
    rule.head_predicate
    for rule in source.rules
    if rule.head_annotation is not None or (rule.head_lower == 0.0 and rule.head_upper == 1.0)
  }
  bottom_visible_reads = tuple(
    sorted(
      {
        (rule.name, clause.predicate)
        for rule in source.rules
        for clause in rule.clauses
        if clause.predicate in bottom_admitting_heads
        and clause.lower == 0.0
        and clause.upper == 1.0
      }
    )
  )
  if bottom_visible_reads:
    names = ", ".join(f"{rule_name!r}:{predicate}" for rule_name, predicate in bottom_visible_reads)
    raise UnsupportedAnnotationSemantics(
      "a rule can admit a missing head key at [0,1] without a PyReason "
      "interval-change event, while a downstream [0,1] body read would see "
      "SRDatalog's relational insertion as a recursive delta; exact support "
      f"needs a logical head-change generation ({names})"
    )

  rule_head_predicates = {rule.head_predicate for rule in source.rules}
  unstable_callback_defaults = tuple(
    sorted(
      {
        (rule.name, clause.predicate)
        for rule in source.rules
        if rule.head_annotation is not None
        for clause in rule.clauses
        if clause.predicate in source.closed_world_predicates
        and clause.lower == 0.0
        and clause.predicate in rule_head_predicates
      }
    )
  )
  if unstable_callback_defaults:
    names = ", ".join(
      f"{rule_name!r}:{predicate}" for rule_name, predicate in unstable_callback_defaults
    )
    raise UnsupportedAnnotationSemantics(
      "a callback aggregate cannot retain a closed-world default member after "
      "the same predicate is derived and that member stops qualifying in "
      f"PyReason's next source snapshot ({names})"
    )

  if stable_demands:
    demanded_rule_indices = {
      witness.identity.rule_index
      for witness in rewrite_for_demands(source, stable_demands).rule_witnesses
    }
    demanded_callbacks = [
      source.rules[index].name
      for index in sorted(demanded_rule_indices)
      if source.rules[index].head_annotation is not None
    ]
    if demanded_callbacks:
      names = ", ".join(repr(name) for name in demanded_callbacks)
      raise UnsupportedAnnotationSemantics(
        "on-demand callback provenance needs a logical aggregate-change event "
        f"that freezes every member of the selected group ({names}); raw "
        "candidate rows are not used as an approximation"
      )

  functions = dict(source.annotation_functions)
  callbacks_by_rule: dict[int, str] = {}
  for rule_index, rule in enumerate(source.rules):
    if rule.head_annotation is None:
      continue
    name = rule.head_annotation
    assert name is not None
    function = functions.get(name)
    if function is None:
      raise UnsupportedAnnotationSemantics(
        f"rule {rule.name!r} uses unregistered annotation {name!r}"
      )
    try:
      python_source = callback_source(function)
    except RuleCompileError as exc:
      raise UnsupportedAnnotationSemantics(
        f"annotation {name!r} cannot be compiled: {exc}"
      ) from exc
    callbacks_by_rule[rule_index] = python_source

  if any(rule.set_static for rule in source.rules):
    raise UnsupportedAnnotationSemantics(
      "set_static rule heads are not yet represented by SRDatalog Core"
    )
  return _build_plan(
    source,
    callbacks_by_rule,
    timesteps=timesteps,
    output_root=Path(output_root),
    provenance_demands=stable_demands,
  )


def _map_sensitive_closed_world_reads(source: SourceProgram) -> frozenset[str]:
  '''Predicates read with no clause component bound by an earlier clause.'''
  result: set[str] = set()
  for rule in source.rules:
    bound_terms: set[str] = set()
    for clause in rule.clauses:
      if (
        clause.predicate in source.closed_world_predicates
        and clause.lower == 0.0
        and all(term not in bound_terms for term in clause.terms)
      ):
        result.add(clause.predicate)
      bound_terms.update(clause.terms)
  return frozenset(result)


def _closed_world_default_reads(source: SourceProgram) -> frozenset[str]:
  '''Predicates whose stored range can be coerced to false while reading.'''
  return frozenset(
    clause.predicate
    for rule in source.rules
    for clause in rule.clauses
    if clause.predicate in source.closed_world_predicates and clause.lower == 0.0
  )


def _overlapping_static_fact_keys(
  source: SourceProgram,
  timesteps: int,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
  '''Find same-key fact schedules for which PyReason's first freeze is observable.'''
  result: set[tuple[str, tuple[str, ...]]] = set()
  for index, left in enumerate(source.facts):
    if not _fact_has_occurrence(left, timesteps):
      continue
    left_end = timesteps if left.static else min(left.end_time, timesteps)
    for right in source.facts[index + 1 :]:
      if (
        left.predicate != right.predicate
        or left.arguments != right.arguments
        or not (left.static or right.static)
      ):
        continue
      if not _fact_has_occurrence(right, timesteps):
        continue
      if (
        left.static
        and right.static
        and left.lower == right.lower
        and left.upper == right.upper
        and left.start_time == right.start_time
        and left.end_time == right.end_time
      ):
        # GraphML readers can report the same immutable attribute through two
        # declaration paths. PyReason's second insertion is an exact no-op, as
        # is SRDatalog's set/lattice deduplication.
        continue
      right_end = timesteps if right.static else min(right.end_time, timesteps)
      if max(left.start_time, right.start_time) <= min(left_end, right_end):
        result.add((left.predicate, left.arguments))
  return tuple(sorted(result))


def _build_plan(
  source: SourceProgram,
  callbacks_by_rule: dict[int, str],
  *,
  timesteps: int,
  output_root: Path,
  provenance_demands: tuple[DemandSeed, ...],
) -> NativePlan:
  from srdatalog import (
    Program,
    Relation,
    Var,
    interval_lattice,
  )

  settings = dict(source.settings)
  arities = _predicate_arities(source)
  predicates = tuple(sorted(arities))
  relation_names = {
    predicate: f"PyReasonRelation{index}" for index, predicate in enumerate(predicates)
  }
  facts_by_predicate: dict[str, list[Any]] = {predicate: [] for predicate in predicates}
  for fact in source.facts:
    facts_by_predicate[fact.predicate].append(fact)

  logical = {}
  seeds = {}
  admission_ranks = {}
  for index, predicate in enumerate(predicates):
    arity = arities[predicate]
    logical[predicate] = Relation(
      relation_names[predicate],
      arity + 3,
      column_types=(int,) * (arity + 3),
      value_spec=interval_lattice(
        key_columns=tuple(range(arity + 1)),
        lower_column=arity + 1,
        upper_column=arity + 2,
      ),
    )
    if facts_by_predicate[predicate]:
      seeds[predicate] = Relation(
        f"PyReasonSeed{index}",
        arity + 4,
        column_types=(int,) * (arity + 4),
        input_file=f"seed_{index}.csv",
      )
      admission_ranks[predicate] = Relation(
        f"PyReasonAdmissionRank{index}",
        arity + 1,
        column_types=(int,) * (arity + 1),
        input_file=f"admission_rank_{index}.csv",
      )

  demanded_rule_indices = {
    witness.identity.rule_index
    for witness in rewrite_for_demands(source, provenance_demands).rule_witnesses
  }
  witness_relations: dict[int, WitnessRelations] = {}
  witness_outputs: list[RuleWitnessOutput] = []
  for rule_index in sorted(demanded_rule_indices):
    relations, output = _rule_witness_relations(source, rule_index)
    witness_relations[rule_index] = relations
    witness_outputs.append(output)

  update_clock = Relation(
    "PyReasonUpdateClock",
    1,
    column_types=(int,),
    input_file="update_clock.csv",
  )
  round_sync = Relation(
    "PyReasonRoundSync",
    1,
    column_types=(int,),
    input_file="round_sync.csv",
  )
  horizon_event = Relation(
    "PyReasonHorizonEvent",
    1,
    column_types=(int,),
    input_file="horizon_event.csv",
  )
  time_leq = Relation(
    "PyReasonTimeLeq",
    2,
    column_types=(int, int),
    input_file="time_leq.csv",
  )
  live_time = Relation(
    "PyReasonLiveTime",
    1,
    column_types=(int,),
  )
  head_edge_domain = (
    Relation(
      "PyReasonActiveEdgeTime",
      3,
      column_types=(int, int, int),
      input_file="active_edge_time.csv",
    )
    if any(len(rule.head_terms) == 2 for rule in source.rules)
    else None
  )
  deltas = {rule.delay for rule in source.rules if rule.delay > 0}
  if timesteps > 0:
    deltas.add(1)
  if settings.get("persistent", False):
    deltas.add(1)
  ticks = {
    delta: Relation(
      f"PyReasonTick{delta}",
      2,
      column_types=(int, int),
      input_file=f"tick_{delta}.csv",
    )
    for delta in sorted(deltas)
  }

  rules: list[Any] = []
  live = Var("pyreason_live_time")
  event = Var("pyreason_horizon_time")
  rules.append(
    (live_time(live) <= time_leq(live, event) & horizon_event(event)).named(
      "PyReasonLiveThroughHorizon"
    )
  )
  for predicate in predicates:
    if predicate not in seeds:
      continue
    arity = arities[predicate]
    arguments = tuple(Var(f"seed_{relation_names[predicate]}_{i}") for i in range(arity))
    time = Var(f"seed_{relation_names[predicate]}_time")
    rank = Var(f"seed_{relation_names[predicate]}_rank")
    lower = Var(f"seed_{relation_names[predicate]}_lower")
    upper = Var(f"seed_{relation_names[predicate]}_upper")
    rules.append(
      (
        logical[predicate](*arguments, time, lower, upper)
        <= seeds[predicate](*arguments, time, rank, lower, upper)
      ).named(f"LoadSeed{relation_names[predicate]}")
    )

  (
    closed_world_rules,
    closed_world_arities,
    closed_world_views,
    closed_world_eligibility,
    closed_world_uses_time_leq,
  ) = _closed_world_ranges(source, arities, predicates, logical, seeds)
  rules.extend(closed_world_rules)

  head_predicates = {rule.head_predicate for rule in source.rules}
  update_predicates = set(head_predicates)
  if settings.get("persistent", False) or timesteps > 0:
    update_predicates.update(predicates)
  rules.extend(_round_bridge_rules(update_predicates, arities, logical, round_sync))
  callback_fd_relations: list[str] = []
  for rule_index, rule in enumerate(source.rules):
    target_name = _target_rule_name(rule.name, rule_index)
    python_source = callbacks_by_rule.get(rule_index)
    try:
      compiled_rule = compile_rule(
        python_source,
        rule,
        rule_index,
        logical,
        admission_ranks,
        ticks,
        head_predicates,
        target_name,
        round_sync,
        update_clock,
        horizon_event,
        closed_world_predicates=source.closed_world_predicates,
        closed_world_views=closed_world_views,
        closed_world_eligibility=closed_world_eligibility,
        head_edge_domain=head_edge_domain,
        witness_relations=witness_relations.get(rule_index),
      )
    except RuleCompileError as exc:
      feature = (
        f"annotation {rule.head_annotation!r}"
        if python_source is not None
        else f"rule {rule.name!r}"
      )
      raise UnsupportedAnnotationSemantics(f"{feature} cannot be compiled: {exc}") from exc
    rules.extend(compiled_rule.preparation)
    rules.append(compiled_rule.materialization)
    rules.extend(compiled_rule.temporal_events)
    callback_fd_relations.extend(compiled_rule.diagnostic_relations)
    if compiled_rule.witnesses:
      rules.extend(compiled_rule.witnesses)
      rules.append(
        _witness_round_bridge(
          compiled_rule.witnesses[-1].head,
          round_sync,
          rule_index,
        )
      )

  if settings.get("persistent", False):
    for predicate in predicates:
      rules.append(
        _frame_rule(
          predicate,
          arities[predicate],
          logical,
          ticks[1],
          live_time,
        )
      )
  if timesteps > 0:
    for predicate in predicates:
      rules.extend(
        _temporal_state_rules(
          predicate,
          arities[predicate],
          logical,
          ticks[1],
          live_time,
          reset_nonpersistent=not settings.get("persistent", False),
        )
      )

  fingerprint = _fingerprint(
    source,
    callbacks_by_rule,
    timesteps,
    provenance_demands,
  )
  data_dir = output_root / f"pyreason_{fingerprint}"
  data_dir.mkdir(parents=True, exist_ok=True)
  symbol_ids = _write_inputs(
    source,
    predicates,
    facts_by_predicate,
    data_dir,
    timesteps,
    ticks,
    closed_world_arities,
    closed_world_uses_time_leq,
    frozenset(closed_world_eligibility),
    write_head_edge_domain=head_edge_domain is not None,
  )
  output_predicates = set(predicates)
  outputs = tuple(
    TemporalIntervalOutput(
      relation=relation_names[predicate],
      predicate=predicate,
      argument_columns=tuple(range(arities[predicate])),
      time_column=arities[predicate],
      lower_column=arities[predicate] + 1,
      upper_column=arities[predicate] + 2,
      suppress_bottom=False,
    )
    for predicate in sorted(output_predicates)
  )
  return NativePlan(
    program=Program(rules=rules),
    project_name=f"PyReason{fingerprint}",
    data_dir=data_dir,
    outputs=outputs,
    symbol_names={identifier: symbol for symbol, identifier in symbol_ids.items()},
    rewriter="pyreason-srdatalog-core-v9",
    witness_outputs=tuple(witness_outputs),
    effective_source=source,
    live_time_relation=live_time.name,
    callback_fd_relations=tuple(callback_fd_relations),
    callback_read_predicates=tuple(
      sorted(
        {
          clause.predicate
          for rule in source.rules
          if rule.head_annotation is not None
          for clause in rule.clauses
        }
      )
    ),
  )


def _rule_witness_relations(
  source: SourceProgram,
  rule_index: int,
) -> tuple[WitnessRelations, RuleWitnessOutput]:
  '''Declare candidate, aggregate-winner, and immutable history relations.'''
  from srdatalog import Relation, max_lower_lattice

  rule = source.rules[rule_index]
  head_arity = len(rule.head_terms)
  variables = _rule_variables(rule)
  head_columns = tuple(range(head_arity))
  head_time_column = head_arity
  rank_column = head_arity + 1
  candidate_lower_column = head_arity + 2
  candidate_upper_column = head_arity + 3
  binding_start = head_arity + 4
  binding_columns = tuple(
    (variable, binding_start + index) for index, variable in enumerate(variables)
  )
  interval_start = binding_start + len(binding_columns)
  body_lower_columns = tuple(interval_start + 2 * index for index in range(len(rule.clauses)))
  body_upper_columns = tuple(column + 1 for column in body_lower_columns)
  arity = interval_start + 2 * len(rule.clauses)
  candidate = Relation(
    f"PyReasonWitness{rule_index}Candidate",
    arity,
    column_types=(int,) * arity,
  )
  winner_arity = head_arity + 4
  winner = Relation(
    f"PyReasonWitness{rule_index}Winner",
    winner_arity,
    column_types=(int,) * winner_arity,
    value_spec=max_lower_lattice(
      key_columns=tuple(range(head_arity + 1)),
      rank_column=head_arity + 1,
      lower_column=head_arity + 2,
      upper_column=head_arity + 3,
    ),
  )
  history = Relation(
    f"PyReasonWitness{rule_index}",
    arity,
    column_types=(int,) * arity,
  )
  output = RuleWitnessOutput(
    relation=history.name,
    rule_index=rule_index,
    rule_name=rule.name,
    head_predicate=rule.head_predicate,
    head_argument_columns=head_columns,
    head_time_column=head_time_column,
    rank_column=rank_column,
    candidate_lower_column=candidate_lower_column,
    candidate_upper_column=candidate_upper_column,
    binding_columns=binding_columns,
    body_lower_columns=body_lower_columns,
    body_upper_columns=body_upper_columns,
    closed_world_clause_positions=tuple(
      index
      for index, clause in enumerate(rule.clauses)
      if clause.predicate in source.closed_world_predicates and clause.lower == 0.0
    ),
  )
  return WitnessRelations(candidate=candidate, winner=winner, history=history), output


def _rule_variables(rule: Any) -> tuple[str, ...]:
  return tuple(
    dict.fromkeys((*rule.head_terms, *(term for clause in rule.clauses for term in clause.terms)))
  )


def _witness_round_bridge(
  witness_head: Any,
  round_clock: Any,
  rule_index: int,
) -> Any:
  '''Keep witness capture in the same source-round SCC as mutable predicates.'''
  # The time column is not necessarily last in the witness tuple.  It is the
  # source head arity, immediately before rank and interval payload columns.
  # Locate it by the stable compiler variable name used by both rule paths.
  time_argument = next(
    argument
    for argument in witness_head.args
    if argument.var_name
    in {
      f"rule_{rule_index}_time",
      f"rule_{rule_index}_next_time",
    }
  )
  return (round_clock(time_argument) <= witness_head).named(
    f"PyReasonWitness{rule_index}RoundBridge"
  )


def _predicate_arities(source: SourceProgram) -> dict[str, int]:
  arities: dict[str, int] = {}

  def admit(predicate: str, arity: int) -> None:
    existing = arities.setdefault(predicate, arity)
    if existing != arity:
      raise UnsupportedAnnotationSemantics(
        f"predicate {predicate!r} is used with arities {existing} and {arity}"
      )

  for fact in source.facts:
    admit(fact.predicate, len(fact.arguments))
  for rule in source.rules:
    admit(rule.head_predicate, len(rule.head_terms))
    for clause in rule.clauses:
      admit(clause.predicate, len(clause.terms))
  for left, right in source.inconsistent_predicates:
    left_arity = arities.get(left)
    right_arity = arities.get(right)
    if left_arity is None and right_arity is None:
      raise UnsupportedAnnotationSemantics(
        f"cannot infer arity for inconsistent predicate pair {(left, right)!r}"
      )
    if left_arity is None:
      assert right_arity is not None
      arities[left] = right_arity
    elif right_arity is None:
      arities[right] = left_arity
    elif left_arity != right_arity:
      raise UnsupportedAnnotationSemantics(
        f"inconsistent predicate pair {(left, right)!r} must have the same arity; "
        f"got {left_arity} and {right_arity}"
      )
  return arities


def _closed_world_ranges(
  source: SourceProgram,
  arities: dict[str, int],
  predicates: tuple[str, ...],
  logical: dict[str, Any],
  seeds: dict[str, Any],
) -> tuple[list[Any], frozenset[int], dict[str, Any], dict[str, Any], bool]:
  """Build a transient closed-world clause-read view.

  A registered predicate is total over its captured node/edge domain and the
  finite time horizon. Missing tuples enter a dedicated interval-lattice read
  relation as ``[0,1]``; actual fact/rule rows flow into that same read view.
  The materialized logical relation stays free of synthetic range tuples, so
  an explicit or reset ``[0,1]`` label remains observable and explainable.

  This is one source program plus a compiler-generated relational view, not a
  second annotated AST.  Eligibility is separate because PyReason's predicate
  map changes the grounding domain. Unsupported nonmonotone first-label
  transitions are rejected before this lowering.
  """
  from srdatalog import Relation, Var, float32_to_u32, interval_lattice

  reads = tuple(
    sorted(
      {
        clause.predicate
        for rule in source.rules
        for clause in rule.clauses
        if clause.predicate in source.closed_world_predicates and clause.lower == 0.0
      }
    )
  )
  if not reads:
    return [], frozenset(), {}, {}, False

  relation_indices = {predicate: index for index, predicate in enumerate(predicates)}
  domain_relations = {
    1: Relation(
      "PyReasonNodeTimeDomain",
      2,
      column_types=(int, int),
      input_file="cwa_node_time_domain.csv",
    ),
    2: Relation(
      "PyReasonEdgeTimeDomain",
      3,
      column_types=(int, int, int),
      input_file="cwa_edge_time_domain.csv",
    ),
  }
  rules: list[Any] = []
  used_arities: set[int] = set()
  views: dict[str, Any] = {}
  eligibility: dict[str, Any] = {}
  bottom_lower = float32_to_u32(0.0)
  bottom_upper = float32_to_u32(1.0)
  time_leq = Relation(
    "PyReasonCwaTimeLeq",
    2,
    column_types=(int, int),
    input_file="cwa_time_leq.csv",
  )

  for predicate in reads:
    arity = arities[predicate]
    if arity not in domain_relations:
      raise UnsupportedAnnotationSemantics(
        f"closed-world predicate {predicate!r} has arity {arity}; "
        "PyReason active-domain reads currently support unary nodes and binary edges"
      )
    used_arities.add(arity)
    index = relation_indices[predicate]
    view = Relation(
      f"PyReasonCwaRead{index}",
      arity + 3,
      column_types=(int,) * (arity + 3),
      value_spec=interval_lattice(
        key_columns=tuple(range(arity + 1)),
        lower_column=arity + 1,
        upper_column=arity + 2,
      ),
    )
    views[predicate] = view
    arguments = tuple(Var(f"cwa_{index}_arg_{column}") for column in range(arity))
    time = Var(f"cwa_{index}_time")
    rules.append(
      (
        view(*arguments, time, bottom_lower, bottom_upper)
        <= domain_relations[arity](*arguments, time)
      ).named(f"PyReasonCwaRange{index}")
    )
    actual_lower = Var(f"cwa_{index}_actual_lower")
    actual_upper = Var(f"cwa_{index}_actual_upper")
    rules.append(
      (
        view(*arguments, time, actual_lower, actual_upper)
        <= logical[predicate](*arguments, time, actual_lower, actual_upper)
      ).named(f"PyReasonCwaActual{index}")
    )
    if predicate in seeds:
      seed_rank = Var(f"cwa_{index}_seed_rank")
      seed_lower = Var(f"cwa_{index}_seed_lower")
      seed_upper = Var(f"cwa_{index}_seed_upper")
      rules.append(
        (
          view(*arguments, time, seed_lower, seed_upper)
          <= seeds[predicate](
            *arguments,
            time,
            seed_rank,
            seed_lower,
            seed_upper,
          )
        ).named(f"PyReasonCwaSeed{index}")
      )

    eligible = Relation(
      f"PyReasonCwaEligible{index}",
      arity + 1,
      column_types=(int,) * (arity + 1),
      input_file=f"cwa_eligible_{index}.csv",
    )
    eligibility[predicate] = eligible
    seen_time = Var(f"cwa_{index}_seen_time")
    eligible_time = Var(f"cwa_{index}_eligible_time")
    lower = Var(f"cwa_{index}_eligible_lower")
    upper = Var(f"cwa_{index}_eligible_upper")
    rules.append(
      (
        eligible(*arguments, eligible_time)
        <= logical[predicate](*arguments, seen_time, lower, upper)
        & time_leq(seen_time, eligible_time)
      ).named(f"PyReasonCwaEligible{index}Derived")
    )

  return (
    rules,
    frozenset(used_arities),
    views,
    eligibility,
    bool(reads),
  )


def _round_bridge_rules(
  predicates: set[str],
  arities: dict[str, int],
  logical: dict[str, Any],
  round_clock: Any,
) -> list[Any]:
  """Put every mutable world predicate in one synchronous rule SCC.

  PyReason grounds every rule against one fixed-point snapshot and applies the
  queued updates only afterward.  The already-present singleton clock is
  semantically inert; the two dependency directions make SRDatalog schedule
  all mutable predicates in the same semi-naive SCC, reproducing those rounds.
  """
  from srdatalog import Var

  rules: list[Any] = []
  for bridge_index, predicate in enumerate(sorted(predicates)):
    arity = arities[predicate]
    arguments = tuple(Var(f"round_bridge_{bridge_index}_arg_{column}") for column in range(arity))
    time = Var(f"round_bridge_{bridge_index}_time")
    lower = Var(f"round_bridge_{bridge_index}_lower")
    upper = Var(f"round_bridge_{bridge_index}_upper")
    rules.append(
      (round_clock(time) <= logical[predicate](*arguments, time, lower, upper)).named(
        f"PyReasonRoundBridge{bridge_index}"
      )
    )
  return rules


def _target_rule_name(source_name: str, rule_index: int) -> str:
  readable = re.sub(r"[^A-Za-z0-9_]+", "_", source_name).strip("_") or "rule"
  return f"PyReasonRule{rule_index}_{readable}"


def _frame_rule(
  predicate: str,
  arity: int,
  logical: Any,
  tick: Any,
  live_time: Any,
) -> Any:
  from srdatalog import Var

  arguments = tuple(Var(f"frame_{predicate}_arg_{i}") for i in range(arity))
  time = Var(f"frame_{predicate}_time")
  next_time = Var(f"frame_{predicate}_next")
  lower = Var(f"frame_{predicate}_lower")
  upper = Var(f"frame_{predicate}_upper")
  return (
    logical[predicate](*arguments, next_time, lower, upper)
    <= logical[predicate](*arguments, time, lower, upper)
    & tick(time, next_time)
    & live_time(next_time)
  ).named(f"Frame{predicate}")


def _temporal_state_rules(
  predicate: str,
  arity: int,
  logical: dict[str, Any],
  tick: Any,
  live_time: Any,
  *,
  reset_nonpersistent: bool,
) -> list[Any]:
  """Lower PyReason's retained label keys and conflict freeze across time.

  Nonpersistent PyReason resets an existing, nonstatic label to visible
  ``[0,1]``; it does not delete the label or its predicate-map membership.
  A crossed lattice value is canonicalized to absorbing top ``[1,0]`` and
  carried forever, modeling PyReason's repaired-unknown static freeze.
  """

  from srdatalog import Var, float32_to_u32
  from srdatalog.dsl import Filter, ScalarCompare, ScalarVar

  arguments = tuple(Var(f"temporal_{predicate}_arg_{i}") for i in range(arity))
  time = Var(f"temporal_{predicate}_time")
  next_time = Var(f"temporal_{predicate}_next")
  lower = Var(f"temporal_{predicate}_lower")
  upper = Var(f"temporal_{predicate}_upper")
  bottom_lower = float32_to_u32(0.0)
  bottom_upper = float32_to_u32(1.0)
  top_lower = float32_to_u32(1.0)
  top_upper = float32_to_u32(0.0)
  crossed = Filter(
    expression=ScalarCompare(
      ">",
      ScalarVar(lower.name),
      ScalarVar(upper.name),
    )
  )
  result: list[Any] = [
    (
      logical[predicate](*arguments, time, top_lower, top_upper)
      <= logical[predicate](*arguments, time, lower, upper) & crossed
    ).named(f"CanonicalConflict{predicate}"),
    (
      logical[predicate](*arguments, next_time, top_lower, top_upper)
      <= logical[predicate](*arguments, time, lower, upper)
      & crossed
      & tick(time, next_time)
      & live_time(next_time)
    ).named(f"FreezeConflict{predicate}"),
  ]
  if reset_nonpersistent:
    result.append(
      (
        logical[predicate](*arguments, next_time, bottom_lower, bottom_upper)
        <= logical[predicate](*arguments, time, lower, upper)
        & tick(time, next_time)
        & live_time(next_time)
      ).named(f"ResetLabel{predicate}")
    )
  return result


def _write_inputs(
  source: SourceProgram,
  predicates: tuple[str, ...],
  facts_by_predicate: dict[str, list[Any]],
  data_dir: Path,
  timesteps: int,
  ticks: dict[int, Any],
  closed_world_arities: frozenset[int],
  closed_world_uses_time_leq: bool,
  closed_world_eligibility: frozenset[str],
  *,
  write_head_edge_domain: bool,
) -> dict[str, int]:
  from srdatalog import float32_to_u32

  symbols = sorted(
    {
      *(argument for fact in source.facts for argument in fact.arguments),
      *source.node_domain,
      *(argument for edge in source.edge_domain for argument in edge),
    }
  )
  symbol_ids = {symbol: index for index, symbol in enumerate(symbols)}
  predicate_indices = {predicate: index for index, predicate in enumerate(predicates)}
  node_births = dict(source.node_birth_times)
  edge_births = {
    (source_node, target_node): birth for source_node, target_node, birth in source.edge_birth_times
  }
  for predicate, facts in facts_by_predicate.items():
    if not facts:
      continue
    admission_ranks = _fact_admission_ranks(facts, timesteps)
    _write_csv(
      data_dir / f"admission_rank_{predicate_indices[predicate]}.csv",
      (
        (
          *(symbol_ids[argument] for argument in arguments),
          rank,
        )
        for arguments, rank in admission_ranks.items()
      ),
    )
    rows = []
    for fact in facts:
      if not _fact_has_occurrence(fact, timesteps):
        continue
      start = fact.start_time
      end = timesteps if fact.static else min(fact.end_time, timesteps)
      for time in range(start, end + 1):
        rows.append(
          (
            *(symbol_ids[argument] for argument in fact.arguments),
            time,
            admission_ranks[fact.arguments],
            float32_to_u32(fact.lower),
            float32_to_u32(fact.upper),
          )
        )
    _write_csv(data_dir / f"seed_{predicate_indices[predicate]}.csv", rows)
  for delta in ticks:
    _write_csv(
      data_dir / f"tick_{delta}.csv",
      ((time, time + delta) for time in range(timesteps - delta + 1)),
    )
  _write_csv(
    data_dir / "update_clock.csv",
    ((time,) for time in _fact_update_times(source, timesteps)),
  )
  _write_csv(
    data_dir / "round_sync.csv",
    ((time,) for time in range(timesteps + 1)),
  )
  max_fact_time = max(
    (min(fact.end_time, timesteps) for fact in source.facts if fact.end_time >= fact.start_time),
    default=0,
  )
  _write_csv(data_dir / "horizon_event.csv", ((max_fact_time,),))
  _write_csv(
    data_dir / "time_leq.csv",
    (
      (live_time, event_time)
      for event_time in range(timesteps + 1)
      for live_time in range(event_time + 1)
    ),
  )
  if write_head_edge_domain:
    _write_csv(
      data_dir / "active_edge_time.csv",
      (
        (symbol_ids[source_node], symbol_ids[target_node], time)
        for source_node, target_node in source.edge_domain
        for time in range(
          edge_births.get((source_node, target_node), 0),
          timesteps + 1,
        )
      ),
    )
  if closed_world_uses_time_leq:
    _write_csv(
      data_dir / "cwa_time_leq.csv",
      (
        (seen_time, eligible_time)
        for seen_time in range(timesteps + 1)
        for eligible_time in range(seen_time, timesteps + 1)
      ),
    )
  for predicate in sorted(closed_world_eligibility):
    arity = _predicate_arities(source)[predicate]
    predicate_facts = tuple(
      fact
      for fact in source.facts
      if fact.predicate == predicate and _fact_has_occurrence(fact, timesteps)
    )
    eligibility_rows: list[tuple[int, ...]] = []
    for time in range(timesteps + 1):
      domain: tuple[tuple[str, ...], ...]
      if arity == 1:
        domain = tuple((node,) for node in source.node_domain if node_births.get(node, 0) <= time)
      elif arity == 2:
        domain = tuple(edge for edge in source.edge_domain if edge_births.get(edge, 0) <= time)
      else:
        continue
      labeled = {fact.arguments for fact in predicate_facts if fact.start_time <= time}
      grounding_domain = tuple(sorted(labeled)) if labeled else domain
      eligibility_rows.extend(
        (
          *(symbol_ids[argument] for argument in arguments),
          time,
        )
        for arguments in grounding_domain
      )
    _write_csv(
      data_dir / f"cwa_eligible_{predicate_indices[predicate]}.csv",
      eligibility_rows,
    )
  if 1 in closed_world_arities:
    _write_csv(
      data_dir / "cwa_node_time_domain.csv",
      (
        (symbol_ids[node], time)
        for node in source.node_domain
        for time in range(node_births.get(node, 0), timesteps + 1)
      ),
    )
  if 2 in closed_world_arities:
    _write_csv(
      data_dir / "cwa_edge_time_domain.csv",
      (
        (symbol_ids[source_node], symbol_ids[target_node], time)
        for source_node, target_node in source.edge_domain
        for time in range(
          edge_births.get((source_node, target_node), 0),
          timesteps + 1,
        )
      ),
    )
  return symbol_ids


def _fact_update_times(source: SourceProgram, timesteps: int) -> tuple[int, ...]:
  """Times at which scheduled EDB writes can make PyReason ground rules.

  The perfect-convergence horizon is handled separately.  Exact ``[0,1]``
  writes can admit a predicate-map key but `_update_*` reports no interval
  change, and repeated static writes are skipped after the first freeze.
  """

  times: set[int] = set()
  for fact in source.facts:
    if not _fact_has_occurrence(fact, timesteps):
      continue
    if fact.lower == 0.0 and fact.upper == 1.0:
      continue
    if fact.static:
      times.add(fact.start_time)
      continue
    times.update(range(fact.start_time, min(fact.end_time, timesteps) + 1))
  return tuple(sorted(times))


def _fact_admission_ranks(
  facts: list[Any],
  timesteps: int,
) -> dict[tuple[str, ...], int]:
  """Rank EDB keys by PyReason predicate-map admission order.

  Graph attribute keys are installed while the world is initialized. Remaining
  fact keys are admitted by first scheduled timestep and effective fact order.
  All raw rows for one key share the rank because callbacks observe one merged
  world interval for that key.
  """

  first_admission: dict[tuple[str, ...], tuple[int, int, int]] = {}
  for source_index, fact in enumerate(facts):
    if not _fact_has_occurrence(fact, timesteps):
      continue
    order = (0, 0, source_index) if fact.graph_attribute else (1, fact.start_time, source_index)
    first_admission[fact.arguments] = min(
      first_admission.get(fact.arguments, order),
      order,
    )
  return {
    # Rank zero is reserved for a strict-max callback's synthetic accumulator
    # initializer.  Real predicate-map admissions retain their order at 1..N.
    arguments: rank + 1
    for rank, (arguments, _) in enumerate(sorted(first_admission.items(), key=lambda item: item[1]))
  }


def _write_csv(path: Path, rows: Any) -> None:
  with path.open("w", newline="") as handle:
    csv.writer(handle).writerows(rows)


def _fingerprint(
  source: SourceProgram,
  callbacks_by_rule: dict[int, str],
  timesteps: int,
  provenance_demands: tuple[DemandSeed, ...],
) -> str:
  payload = {
    "compiler": "pyreason-srdatalog-core-v9",
    "facts": [asdict(fact) for fact in source.facts],
    "rules": [asdict(rule) for rule in source.rules],
    "callback_sources": callbacks_by_rule,
    "settings": source.settings,
    "closed_world_predicates": sorted(source.closed_world_predicates),
    "node_domain": source.node_domain,
    "edge_domain": source.edge_domain,
    "node_birth_times": source.node_birth_times,
    "edge_birth_times": source.edge_birth_times,
    "reorder_clauses_node_first": source.reorder_clauses_node_first,
    "inconsistent_predicates": source.inconsistent_predicates,
    "provenance_demands": [asdict(demand) for demand in provenance_demands],
    "timesteps": timesteps,
  }
  encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
  return hashlib.sha256(encoded).hexdigest()[:16]
