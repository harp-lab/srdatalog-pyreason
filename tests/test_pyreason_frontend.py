import csv
from pathlib import Path

import pytest

from minimal_vulreasoner.annotation_fn import paired_minimum_bounds_ann_fn
from minimal_vulreasoner.example_config import (
  END_TIME,
  INITIAL_NODE,
  KG_FILE,
  RULES_FILE,
  WORKFLOW,
)
from srdatalog.ir.hir import compile_to_hir
from srdatalog.pyreason import (
  RewriteRejected,
  SourceFact,
  SourceProgram,
  SourceRule,
  try_rewrite,
)


def _source(*, branch: bool = False, register_annotation: bool = True) -> SourceProgram:
  with RULES_FILE.open(newline='') as handle:
    rows = tuple(csv.DictReader(handle))
  rules = tuple(
    SourceRule(
      text=row['rule_text'],
      name=row['name'],
      head_predicate='analystAt',
      head_terms=('CB2',),
      head_annotation='paired_minimum_bounds_ann_fn',
      delay=1,
      clauses=(),
      infer_edges=True,
    )
    for row in rows
  )
  facts = [
    SourceFact(
      text=f'hasLabel({block},{label})',
      name=f'label-{block}',
      predicate='hasLabel',
      arguments=(block, label),
      lower=1.0,
      upper=1.0,
      start_time=0,
      end_time=0,
      static=True,
    )
    for block, label in WORKFLOW
  ]
  facts.append(
    SourceFact(
      text=f'analystAt({INITIAL_NODE})',
      name='initial-control',
      predicate='analystAt',
      arguments=(INITIAL_NODE,),
      lower=1.0,
      upper=1.0,
      start_time=0,
      end_time=1,
      static=False,
    )
  )
  for index in range(len(WORKFLOW) - 1):
    source, target = WORKFLOW[index][0], WORKFLOW[index + 1][0]
    facts.append(
      SourceFact(
        text=f'stepFrom({source},{target})',
        name=f'edge-{index}',
        predicate='stepFrom',
        arguments=(source, target),
        lower=1.0,
        upper=1.0,
        start_time=index + 1,
        end_time=index + 2,
        static=False,
      )
    )
  if branch:
    facts.append(
      SourceFact(
        text='stepFrom(b1,b3)',
        name='branch',
        predicate='stepFrom',
        arguments=('b1', 'b3'),
        lower=1.0,
        upper=1.0,
        start_time=1,
        end_time=2,
        static=False,
      )
    )
  annotations = (
    (('paired_minimum_bounds_ann_fn', paired_minimum_bounds_ann_fn),)
    if register_annotation
    else ()
  )
  return SourceProgram(
    rules=rules,
    facts=tuple(facts),
    graphml_path=str(KG_FILE),
    closed_world_predicates=frozenset({'analystAt'}),
    annotation_functions=annotations,
  )


def test_annotation_owned_rewriter_lowers_to_candidate_program(tmp_path: Path) -> None:
  plan = try_rewrite(_source(), timesteps=END_TIME, output_root=tmp_path)
  assert plan is not None
  assert plan.rewriter == 'minimal-vulreasoner/paired-minimum-v1'
  hir = compile_to_hir(plan.program)
  decls = {decl.rel_name: decl for decl in hir.relation_decls}
  candidates = [decl for name, decl in decls.items() if name.endswith('Candidate')]
  assert len(candidates) == 6
  for decl in candidates:
    assert decl.value_spec is not None
    assert decl.value_spec.join.value == 'max-lower-select'
  assert decls['AnalystAt'].value_spec is not None
  assert decls['AnalystAt'].value_spec.join.value == 'interval-intersection'


def test_unregistered_annotation_does_not_select_application_rewriter(tmp_path: Path) -> None:
  assert try_rewrite(
    _source(register_annotation=False),
    timesteps=END_TIME,
    output_root=tmp_path,
  ) is None


def test_claimed_near_miss_is_rejected_without_fallback(tmp_path: Path) -> None:
  with pytest.raises(RewriteRejected, match='workflow branches'):
    try_rewrite(_source(branch=True), timesteps=END_TIME, output_root=tmp_path)
