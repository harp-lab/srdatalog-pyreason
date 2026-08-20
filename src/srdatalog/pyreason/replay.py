'''On-demand instrumented replay for PyReason why-provenance.

The base run carries only interval values.  A provenance query first performs
the finite backward demand rewrite, recompiles the same source program with
ordinary candidate relations for the reachable source rules, and executes that
instrumented program.  Candidate tuples retain historical body interval
versions under set semantics; no physical GPU tuple identity is exposed.
'''

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .compiler import _effective_source, compile_source
from .demand import DemandSeed, rewrite_for_demands
from .explain import ExplanationGraph, explain_demands
from .model import RuleCandidateRow, SourceProgram, TemporalIntervalRow
from .runtime import execute


@dataclass(frozen=True)
class DemandReplay:
  '''Result of one demand-transformed provenance execution.'''

  graph: ExplanationGraph
  candidates: tuple[RuleCandidateRow, ...]
  metadata: dict[str, object]


def replay_demands(
  source: SourceProgram,
  materialized_rows: Iterable[TemporalIntervalRow],
  seeds: Iterable[DemandSeed],
  *,
  timesteps: int,
  output_root: str | Path,
  cache_base: str | Path,
  jobs: int,
  timeout: float,
) -> DemandReplay:
  '''Run provenance instrumentation only for the reachable demanded rules.'''
  roots = tuple(dict.fromkeys(seeds))
  rows = tuple(materialized_rows)
  source = _effective_source(source)
  rewrite = rewrite_for_demands(source, roots)
  demanded_rule_indices = {witness.identity.rule_index for witness in rewrite.rule_witnesses}
  if not demanded_rule_indices:
    return DemandReplay(
      graph=explain_demands(source, rows, roots, rule_candidates=()),
      candidates=(),
      metadata={'provenance_replay': 'not-needed'},
    )

  plan = compile_source(
    source,
    timesteps=timesteps,
    output_root=output_root,
    provenance_demands=roots,
  )
  result = execute(
    plan,
    cache_base=cache_base,
    jobs=jobs,
    timeout=timeout,
  )
  candidates = tuple(
    sorted(
      (
        candidate
        for candidate in result.witness_rows
        if candidate.rule_index in demanded_rule_indices
      ),
      key=lambda candidate: candidate.witness_key,
    )
  )
  return DemandReplay(
    graph=explain_demands(
      source,
      rows,
      roots,
      rule_candidates=candidates,
    ),
    candidates=candidates,
    metadata={
      **result.metadata,
      'provenance_replay': 'demand-instrumented-srdatalog',
      'demanded_rule_indices': sorted(demanded_rule_indices),
      'candidate_count': len(candidates),
    },
  )
