from pathlib import Path

from srdatalog import compile_to_hir
from srdatalog.dsl import Atom, Rule
from srdatalog.pyreason import (
  DemandSeed,
  SourceClause,
  SourceFact,
  SourceProgram,
  SourceRule,
  compile_source,
)


def _fact(predicate: str, lower: float = 1.0, upper: float = 1.0) -> SourceFact:
  return SourceFact(
    text=f'{predicate}(a):[{lower},{upper}]',
    name=f'{predicate}-a',
    predicate=predicate,
    arguments=('a',),
    lower=lower,
    upper=upper,
    start_time=0,
    end_time=0,
    static=True,
  )


def _rule(
  name: str,
  head: str,
  body: str,
  *,
  head_interval: tuple[float, float] = (1.0, 1.0),
  body_interval: tuple[float, float] = (1.0, 1.0),
) -> SourceRule:
  return SourceRule(
    text=f'{head}(X):[{head_interval[0]},{head_interval[1]}] <- {body}(X)',
    name=name,
    head_predicate=head,
    head_terms=('X',),
    head_annotation=None,
    head_lower=head_interval[0],
    head_upper=head_interval[1],
    delay=0,
    clauses=(SourceClause(body, ('X',), *body_interval),),
  )


def _compile_with_provenance(
  source: SourceProgram,
  demand: DemandSeed,
  tmp_path: Path,
):
  plan = compile_source(
    source,
    timesteps=demand.time,
    output_root=tmp_path,
    provenance_demands=(demand,),
  )
  assert plan is not None
  return plan


def _witness_output(plan, rule_index: int):
  matches = [output for output in plan.witness_outputs if output.rule_index == rule_index]
  assert len(matches) == 1
  return matches[0]


def _witness_rule(plan, relation_name: str) -> Rule:
  matches = [rule for rule in plan.program.rules if rule.head.rel == relation_name]
  assert len(matches) == 1
  return matches[0]


def _assert_historical_candidate_relation(
  plan,
  *,
  rule_index: int,
  body_clauses: int,
  closed_world_clause_positions: tuple[int, ...],
) -> None:
  output = _witness_output(plan, rule_index)
  history_relation = next(
    relation for relation in plan.program.relations if relation.name == output.relation
  )
  candidate_relation = next(
    relation
    for relation in plan.program.relations
    if relation.name == f'{output.relation}Candidate'
  )
  winner_relation = next(
    relation for relation in plan.program.relations if relation.name == f'{output.relation}Winner'
  )
  candidate_rule = _witness_rule(plan, candidate_relation.name)
  winner_rule = _witness_rule(plan, winner_relation.name)
  history_rule = _witness_rule(plan, output.relation)

  # Candidate rows are an ordinary monotone set, not another interval-lattice
  # projection.  An earlier body version must remain after the logical row is
  # narrowed or crosses.
  assert candidate_relation.value_spec is None
  assert history_relation.value_spec is None
  assert winner_relation.value_spec is not None
  assert winner_relation.value_spec.join.value == 'max-lower-select'
  assert len(output.body_lower_columns) == body_clauses
  assert len(output.body_upper_columns) == body_clauses
  assert output.closed_world_clause_positions == closed_world_clause_positions
  for column in (*output.body_lower_columns, *output.body_upper_columns):
    assert 0 <= column < history_relation.arity
    assert history_rule.head.args[column].var_name is not None

  # Raw candidates and the selected state are separate.  History joins them
  # on the aggregate winner key, so losing callback candidates cannot appear
  # as head derivations while an earlier winner version remains immutable.
  assert any(
    isinstance(clause, Atom) and clause.rel == candidate_relation.name
    for clause in history_rule.body
  )
  assert any(
    isinstance(clause, Atom) and clause.rel == winner_relation.name for clause in history_rule.body
  )
  assert candidate_rule.body == winner_rule.body

  # The witness rule must observe the same semi-naive snapshots as the logical
  # rule.  If it were stratified after the logical SCC, only the final repaired
  # interval would remain and both regressions below would return.
  hir = compile_to_hir(plan.program)
  witness_stratum = next(
    stratum
    for stratum in hir.strata
    if stratum.is_recursive and output.relation in stratum.scc_members
  )
  assert 'PyReasonRoundSync' in witness_stratum.scc_members
  witness_decl = next(
    declaration for declaration in hir.relation_decls if declaration.rel_name == output.relation
  )
  assert witness_decl.value_spec is None
  assert any(
    isinstance(clause, Atom) and clause.rel == 'PyReasonRoundSync' for clause in candidate_rule.body
  )


def test_cwa_witness_survives_later_positive_narrowing(tmp_path: Path) -> None:
  source = SourceProgram(
    rules=(
      _rule(
        'safe-by-default',
        'safe',
        'blocked',
        body_interval=(0.0, 0.0),
      ),
      _rule('derive-blocked-later', 'blocked', 'seed'),
    ),
    facts=(_fact('seed'),),
    graphml_path=None,
    closed_world_predicates=frozenset({'blocked'}),
    annotation_functions=(),
    node_domain=('a',),
  )

  plan = _compile_with_provenance(
    source,
    DemandSeed('safe', ('a',), 0),
    tmp_path,
  )

  _assert_historical_candidate_relation(
    plan,
    rule_index=0,
    body_clauses=1,
    closed_world_clause_positions=(0,),
  )


def test_body_witness_survives_later_crossed_interval(tmp_path: Path) -> None:
  source = SourceProgram(
    rules=(
      _rule('copy-consistent-mid', 'result', 'mid'),
      _rule('derive-mid-high', 'mid', 'seed'),
      _rule('derive-stage', 'stage', 'seed'),
      _rule('derive-trigger', 'trigger', 'stage'),
      _rule(
        'cross-mid-later',
        'mid',
        'trigger',
        head_interval=(0.0, 0.2),
      ),
    ),
    facts=(_fact('seed'),),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(),
    node_domain=('a',),
  )

  plan = _compile_with_provenance(
    source,
    DemandSeed('result', ('a',), 0),
    tmp_path,
  )

  _assert_historical_candidate_relation(
    plan,
    rule_index=0,
    body_clauses=1,
    closed_world_clause_positions=(),
  )


def test_ordinary_materialization_has_no_witness_payload(tmp_path: Path) -> None:
  source = SourceProgram(
    rules=(_rule('derive-result', 'result', 'seed'),),
    facts=(_fact('seed'),),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(),
    node_domain=('a',),
  )

  plan = compile_source(source, timesteps=0, output_root=tmp_path)

  assert plan.witness_outputs == ()
  assert not any(relation.name.startswith('PyReasonWitness') for relation in plan.program.relations)
