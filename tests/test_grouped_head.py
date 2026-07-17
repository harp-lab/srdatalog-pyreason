from srdatalog import (
  Program,
  Relation,
  Var,
  compile_to_hir,
  interval_lattice,
  max_lower_lattice,
)
from srdatalog.dsl import Filter


def _interval_relation(name: str, arity: int) -> Relation:
  return Relation(
    name,
    arity + 2,
    value_spec=interval_lattice(
      key_columns=tuple(range(arity)),
      lower_column=arity,
      upper_column=arity + 1,
    ),
  )


def test_grouped_head_expands_for_arbitrary_relation() -> None:
  key, rank, lower, upper = (Var(name) for name in ('key', 'rank', 'lower', 'upper'))
  source = Relation('AnyWitness', 4)
  result = _interval_relation('AnyResult', 1)
  rule = (
    (result(key, lower, upper) <= source(key, rank, lower, upper))
    .named('Choose')
    .with_grouped_head(
      group_args=(key,),
      value_args=(rank, lower, upper),
      value_spec=max_lower_lattice(
        key_columns=(0,),
        rank_column=1,
        lower_column=2,
        upper_column=3,
      ),
    )
  )

  hir = compile_to_hir(Program(rules=[rule]))

  generated = [decl for decl in hir.relation_decls if decl.is_generated]
  assert len(generated) == 1
  assert generated[0].value_spec is not None
  assert generated[0].value_spec.join.value == 'max-lower-select'
  names = [rule.name for stratum in hir.strata for rule in stratum.stratum_rules]
  assert names == ['Choose__group', 'Choose__project']


def test_grouped_recursive_rule_keeps_every_delta_source() -> None:
  key = Var('key')
  left_lower, left_upper = Var('left_lower'), Var('left_upper')
  right_lower, right_upper = Var('right_lower'), Var('right_upper')
  rank, result_lower, result_upper = (
    Var(name) for name in ('rank', 'result_lower', 'result_upper')
  )
  seed = Relation('SeedState', 3)
  witness = Relation('WitnessInput', 4)
  state = _interval_relation('StateValue', 1)
  base = (
    state(key, result_lower, result_upper)
    <= seed(key, result_lower, result_upper)
  ).named('Base')
  recursive = (
    (
      state(key, result_lower, result_upper)
      <= state(key, left_lower, left_upper)
      & state(key, right_lower, right_upper)
      & witness(key, rank, result_lower, result_upper)
    )
    .named('RecursiveChoose')
    .with_grouped_head(
      group_args=(key,),
      value_args=(rank, result_lower, result_upper),
      value_spec=max_lower_lattice(
        key_columns=(0,),
        rank_column=1,
        lower_column=2,
        upper_column=3,
      ),
    )
  )

  hir = compile_to_hir(Program(rules=[base, recursive]))
  grouped = next(
    stratum
    for stratum in hir.strata
    if any(
      variant.original_rule.name == 'RecursiveChoose__group'
      for variant in stratum.recursive_variants
    )
  )
  variants = [
    variant
    for variant in grouped.recursive_variants
    if variant.original_rule.name == 'RecursiveChoose__group'
  ]
  assert [variant.delta_idx for variant in variants] == [0, 1]
  assert [[version.value for version in variant.clause_versions] for variant in variants] == [
    ['DELTA', 'FULL', 'FULL'],
    ['FULL', 'DELTA', 'FULL'],
  ]


def test_grouped_head_finalize_filter_runs_after_generated_state() -> None:
  key, rank, lower, upper = (Var(name) for name in ('key', 'rank', 'lower', 'upper'))
  source = Relation('FilteredWitness', 4)
  result = _interval_relation('FilteredResult', 1)
  rule = (
    (result(key, lower, upper) <= source(key, rank, lower, upper))
    .named('ChooseValid')
    .with_grouped_head(
      group_args=(key,),
      value_args=(rank, lower, upper),
      value_spec=max_lower_lattice(
        key_columns=(0,),
        rank_column=1,
        lower_column=2,
        upper_column=3,
      ),
      finalize_filters=(
        Filter(vars=(lower.name, upper.name), code='return lower <= upper;'),
      ),
    )
  )

  hir = compile_to_hir(Program(rules=[rule]))
  projection = next(
    candidate
    for stratum in hir.strata
    for candidate in stratum.stratum_rules
    if candidate.name == 'ChooseValid__project'
  )
  assert len(projection.body) == 2
  assert isinstance(projection.body[1], Filter)
