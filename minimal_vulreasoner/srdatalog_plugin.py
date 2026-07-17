'''SRDatalog compiler plugin for the extended VulReasoner annotation.'''

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from srdatalog.pyreason import (
  NativePlan,
  RewriteRejected,
  SourceProgram,
  TemporalIntervalOutput,
)

try:
  from .analyst_rule_loader import parse_analyst_rule
  from .graphml_ingest import emit_minimal_dataset
  from .srdatalog_query import build_analyst_program
except ImportError:
  from analyst_rule_loader import parse_analyst_rule  # type: ignore[no-redef]
  from graphml_ingest import emit_minimal_dataset  # type: ignore[no-redef]
  from srdatalog_query import build_analyst_program  # type: ignore[no-redef]

ANNOTATION_FUNCTION = 'paired_minimum_bounds_ann_fn'


class VulReasonerRewriter:
  '''Own the application-specific recognition and lossless relational rewrite.'''

  name = 'minimal-vulreasoner/paired-minimum-v1'

  def claims(self, source: SourceProgram) -> bool:
    return any(rule.head_annotation == ANNOTATION_FUNCTION for rule in source.rules)

  def rewrite(
    self,
    source: SourceProgram,
    *,
    timesteps: int,
    output_root: str | Path,
  ) -> NativePlan:
    if timesteps < 1:
      raise RewriteRejected('VulReasoner rewriting requires a finite positive horizon')
    if source.graphml_path is None:
      raise RewriteRejected(
        'VulReasoner rewriting requires load_graphml(); an arbitrary in-memory '
        'graph cannot preserve GraphML connector rank'
      )
    unsupported_cwa = set(source.closed_world_predicates) - {'analystAt'}
    if unsupported_cwa:
      raise RewriteRejected(
        'the native rewrite only supports closed-world analystAt, not '
        + ', '.join(sorted(unsupported_cwa))
      )
    if any(rule.head_annotation != ANNOTATION_FUNCTION for rule in source.rules):
      raise RewriteRejected(
        'a VulReasoner program cannot mix paired provenance rules with other '
        'annotation/head semantics'
      )

    try:
      specs = tuple(parse_analyst_rule(rule.text, rule.name) for rule in source.rules)
      connector_predicates = tuple(spec.connector_predicate for spec in specs)
      if len(connector_predicates) != len(set(connector_predicates)):
        raise ValueError('duplicate connector predicates')
      workflow, initial_node = _derive_workflow(source)
    except ValueError as exc:
      raise RewriteRejected(
        'program uses paired provenance but is outside the established '
        f'VulReasoner rewrite: {exc}'
      ) from exc

    fingerprint = _fingerprint(source, timesteps)
    data_dir = Path(output_root) / f'vulreasoner_{fingerprint}'
    emit_minimal_dataset(
      source.graphml_path,
      data_dir,
      workflow=workflow,
      initial_node=initial_node,
      end_time=timesteps,
      connector_predicates=connector_predicates,
    )
    symbols = json.loads((data_dir / 'symbols.json').read_text())
    return NativePlan(
      program=build_analyst_program(specs),
      project_name='VulReasonerPlan',
      data_dir=data_dir,
      outputs=(
        TemporalIntervalOutput(
          relation='AnalystAt',
          predicate='analystAt',
          argument_columns=(0,),
          time_column=1,
          lower_column=2,
          upper_column=3,
        ),
      ),
      symbol_names={int(identifier): symbol for symbol, identifier in symbols.items()},
      rewriter=self.name,
    )


def _derive_workflow(source: SourceProgram) -> tuple[tuple[tuple[str, str], ...], str]:
  labels: dict[str, str] = {}
  for fact in (fact for fact in source.facts if fact.predicate == 'hasLabel'):
    if len(fact.arguments) != 2 or not fact.static or (fact.lower, fact.upper) != (1.0, 1.0):
      raise ValueError('hasLabel facts must be static binary [1,1] facts')
    block, label = fact.arguments
    if block in labels:
      raise ValueError(f'workflow block {block!r} has multiple labels')
    labels[block] = label

  seeds = [fact for fact in source.facts if fact.predicate == 'analystAt']
  if len(seeds) != 1:
    raise ValueError('expected exactly one analystAt seed fact')
  seed = seeds[0]
  if (
    len(seed.arguments) != 1
    or seed.static
    or seed.start_time != 0
    or seed.end_time != 1
    or (seed.lower, seed.upper) != (1.0, 1.0)
  ):
    raise ValueError('analystAt seed must be unary [1,1] on inclusive time [0,1]')
  initial = seed.arguments[0]
  if initial not in labels:
    raise ValueError('analystAt seed has no hasLabel fact')

  steps = [fact for fact in source.facts if fact.predicate == 'stepFrom']
  outgoing = {}
  for fact in steps:
    if (
      len(fact.arguments) != 2
      or fact.static
      or (fact.lower, fact.upper) != (1.0, 1.0)
    ):
      raise ValueError('stepFrom facts must be timed binary [1,1] facts')
    source_block, _ = fact.arguments
    if source_block in outgoing:
      raise ValueError(f'workflow branches at {source_block!r}; chain adapter is required')
    outgoing[source_block] = fact

  ordered = [initial]
  seen = {initial}
  while ordered[-1] in outgoing:
    fact = outgoing[ordered[-1]]
    expected_start = len(ordered)
    if fact.start_time != expected_start or fact.end_time != expected_start + 1:
      raise ValueError(
        f'stepFrom {fact.arguments!r} must be active on '
        f'[{expected_start},{expected_start + 1}]'
      )
    target = fact.arguments[1]
    if target in seen:
      raise ValueError('workflow contains a cycle')
    if target not in labels:
      raise ValueError(f'workflow target {target!r} has no hasLabel fact')
    ordered.append(target)
    seen.add(target)
  if len(steps) != len(ordered) - 1 or set(labels) != seen:
    raise ValueError('hasLabel and stepFrom facts must describe one complete chain')
  return tuple((block, labels[block]) for block in ordered), initial


def _fingerprint(source: SourceProgram, timesteps: int) -> str:
  payload = {
    'graphml': str(Path(source.graphml_path or '').resolve()),
    'timesteps': timesteps,
    'rules': [(rule.text, rule.name) for rule in source.rules],
    'facts': [
      (
        fact.text,
        fact.name,
        fact.start_time,
        fact.end_time,
        fact.static,
      )
      for fact in source.facts
      if fact.predicate in {'hasLabel', 'analystAt', 'stepFrom'}
    ],
  }
  return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


REWRITER = VulReasonerRewriter()
