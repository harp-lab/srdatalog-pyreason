from dataclasses import replace
from pathlib import Path

import pytest

from srdatalog import Program, Relation, Var, compile_to_hir
from srdatalog.pyreason import (
  NativePlan,
  RewriteRejected,
  SourceFact,
  SourceProgram,
  SourceRule,
  register_annotation_rewriter,
  try_rewrite,
)


def toy_annotation(*_args: object) -> tuple[float, float]:
  return 1.0, 1.0


class ToyRewriter:
  name = 'test/toy-copy'

  def claims(self, source: SourceProgram) -> bool:
    return any(rule.head_annotation == toy_annotation.__name__ for rule in source.rules)

  def rewrite(
    self,
    source: SourceProgram,
    *,
    timesteps: int,
    output_root: str | Path,
  ) -> NativePlan:
    if timesteps < 0:
      raise RewriteRejected('toy rewrite requires a finite horizon')
    if any(rule.head_predicate != 'output' for rule in source.rules):
      raise RewriteRejected('toy rewrite only accepts output heads')

    item = Var('item')
    input_relation = Relation('Input', 1)
    output_relation = Relation('Output', 1)
    program = Program(
      rules=[
        (output_relation(item) <= input_relation(item)).named('Copy'),
      ]
    )
    return NativePlan(
      program=program,
      project_name='ToyCopy',
      data_dir=Path(output_root),
      outputs=(),
      symbol_names={},
      rewriter=self.name,
    )


REWRITER = ToyRewriter()
register_annotation_rewriter(toy_annotation, REWRITER)


def _source(*, annotation: object | None = toy_annotation) -> SourceProgram:
  functions = () if annotation is None else ((toy_annotation.__name__, annotation),)
  return SourceProgram(
    rules=(
      SourceRule(
        text='output(x):toy_annotation <- input(x)',
        name='copy',
        head_predicate='output',
        head_terms=('x',),
        head_annotation=toy_annotation.__name__,
        delay=0,
        clauses=(),
      ),
    ),
    facts=(
      SourceFact(
        text='input(a)',
        name='seed',
        predicate='input',
        arguments=('a',),
        lower=1.0,
        upper=1.0,
        start_time=0,
        end_time=0,
        static=True,
      ),
    ),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=functions,
  )


def test_registered_callable_selects_application_rewriter(tmp_path: Path) -> None:
  plan = try_rewrite(_source(), timesteps=2, output_root=tmp_path)

  assert plan is not None
  assert plan.rewriter == REWRITER.name
  hir = compile_to_hir(plan.program)
  assert [rule.name for stratum in hir.strata for rule in stratum.stratum_rules] == [
    'Copy'
  ]


def test_unregistered_callable_does_not_select_rewriter(tmp_path: Path) -> None:
  def unregistered_annotation(*_args: object) -> tuple[float, float]:
    return 0.0, 1.0

  assert try_rewrite(
    _source(annotation=unregistered_annotation),
    timesteps=2,
    output_root=tmp_path,
  ) is None


def test_claimed_invalid_source_is_rejected_without_fallback(tmp_path: Path) -> None:
  invalid = _source()
  invalid = replace(
    invalid,
    rules=(replace(invalid.rules[0], head_predicate='unsupported'),),
  )

  with pytest.raises(RewriteRejected, match='only accepts output heads'):
    try_rewrite(invalid, timesteps=2, output_root=tmp_path)
