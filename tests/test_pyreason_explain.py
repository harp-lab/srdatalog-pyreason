from srdatalog.pyreason.demand import DemandSeed
from srdatalog.pyreason.explain import explain_demands
from srdatalog.pyreason.model import (
  RuleCandidateRow,
  SourceClause,
  SourceFact,
  SourceProgram,
  SourceRule,
  TemporalIntervalRow,
)


def _rule(
  name: str,
  head: str,
  body: str,
  *,
  lower: float = 1.0,
  upper: float = 1.0,
) -> SourceRule:
  return SourceRule(
    text=f'{head}(X):[{lower},{upper}] <- {body}(X)',
    name=name,
    head_predicate=head,
    head_terms=('X',),
    head_annotation=None,
    head_lower=lower,
    head_upper=upper,
    delay=0,
    clauses=(SourceClause(body, ('X',), 1.0, 1.0),),
  )


def test_explanation_recursively_follows_only_demanded_chain() -> None:
  source = SourceProgram(
    rules=(
      _rule('target-from-mid', 'target', 'mid', lower=0.0, upper=0.2),
      _rule('mid-from-seed', 'mid', 'seed'),
      _rule('unrelated', 'noise', 'other'),
    ),
    facts=(SourceFact('seed(a)', 'seed-a', 'seed', ('a',), 1, 1, 0, 0, True),),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(),
  )
  rows = (
    TemporalIntervalRow('seed', ('a',), 0, 1, 1),
    TemporalIntervalRow('mid', ('a',), 0, 1, 1),
    TemporalIntervalRow('target', ('a',), 0, 0, 1, inconsistent=True),
  )

  graph = explain_demands(source, rows, (DemandSeed('target', ('a',), 0),))

  assert [atom.demand.predicate for atom in graph.atoms] == ['target', 'mid', 'seed']
  target = graph.atom(DemandSeed('target', ('a',), 0))
  assert target.rules[0].rule_name == 'target-from-mid'
  assert target.rules[0].witness_key == (
    0,
    'target',
    ('a',),
    0,
    (('X', 'a'),),
    ((('X', 'a'),),),
    0.0,
    0.2,
    ((1, 'mid', ('a',)),),
  )
  assert target.facts == ()
  seed = graph.atom(DemandSeed('seed', ('a',), 0))
  assert seed.facts[0].fact_name == 'seed-a'
  assert all(atom.demand.predicate != 'noise' for atom in graph.atoms)


def test_explanation_uses_closed_world_false_rows_on_demand() -> None:
  rule = SourceRule(
    text='safe(X) <- dom(X), ~blocked(X)',
    name='safe-by-default',
    head_predicate='safe',
    head_terms=('X',),
    head_annotation=None,
    head_lower=1.0,
    head_upper=1.0,
    delay=0,
    clauses=(
      SourceClause('dom', ('X',), 1.0, 1.0),
      SourceClause('blocked', ('X',), 0.0, 0.0),
    ),
  )
  source = SourceProgram(
    rules=(rule,),
    facts=(),
    graphml_path=None,
    closed_world_predicates=frozenset({'blocked'}),
    annotation_functions=(),
    node_domain=('a',),
  )
  rows = (
    TemporalIntervalRow('dom', ('a',), 0, 1, 1),
    TemporalIntervalRow('safe', ('a',), 0, 1, 1),
  )

  graph = explain_demands(source, rows, (DemandSeed('safe', ('a',), 0),))

  safe = graph.atom(DemandSeed('safe', ('a',), 0))
  assert safe.rules[0].clauses[1].as_trace_cell() == ['a']
  blocked = graph.atom(DemandSeed('blocked', ('a',), 0))
  assert blocked.closed_world_assumptions[0].witness_key == (
    'closed-world',
    'blocked',
    ('a',),
    0,
    0.0,
    1.0,
    'closed-world-default',
  )


def test_explanation_traverses_an_ipl_witness_without_tuple_ids() -> None:
  source = SourceProgram(
    rules=(),
    facts=(SourceFact('p(a)', 'p-a', 'p', ('a',), 0.2, 0.4, 0, 0, True),),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(),
    inconsistent_predicates=(('p', 'q'),),
  )
  rows = (
    TemporalIntervalRow('p', ('a',), 0, 0.2, 0.4),
    TemporalIntervalRow('q', ('a',), 0, 0.6, 0.8),
  )

  graph = explain_demands(source, rows, (DemandSeed('q', ('a',), 0),))

  q = graph.atom(DemandSeed('q', ('a',), 0))
  assert q.ipls[0].source == DemandSeed('p', ('a',), 0)
  assert q.ipls[0].witness_key == (
    'ipl',
    0,
    0,
    DemandSeed('p', ('a',), 0),
    0.6,
    0.8,
  )
  assert graph.atom(DemandSeed('p', ('a',), 0)).facts[0].fact_name == 'p-a'


def test_rule_explanation_groups_all_clause_entities_for_one_head() -> None:
  rule = SourceRule(
    text='answer(X) <- edge(X,Y)',
    name='all-edges',
    head_predicate='answer',
    head_terms=('X',),
    head_annotation=None,
    head_lower=1.0,
    head_upper=1.0,
    delay=0,
    clauses=(SourceClause('edge', ('X', 'Y'), 1.0, 1.0),),
  )
  source = SourceProgram(
    rules=(rule,),
    facts=(),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(),
  )
  rows = (
    TemporalIntervalRow('edge', ('a', 'b'), 0, 1.0, 1.0),
    TemporalIntervalRow('edge', ('a', 'c'), 0, 1.0, 1.0),
    TemporalIntervalRow('answer', ('a',), 0, 1.0, 1.0),
  )

  graph = explain_demands(source, rows, (DemandSeed('answer', ('a',), 0),))

  answer = graph.atom(DemandSeed('answer', ('a',), 0))
  assert len(answer.rules) == 1
  assert answer.rules[0].clauses[0].as_trace_cell() == [('a', 'b'), ('a', 'c')]
  assert answer.rules[0].groundings == (
    (('X', 'a'), ('Y', 'b')),
    (('X', 'a'), ('Y', 'c')),
  )


def _max_first_clause(
  annotations,
  weights,
  qualified_nodes,
  qualified_edges,
  clause_labels,
  clause_variables,
):
  best = max(annotations[0], key=lambda item: item.lower)
  return best.lower, best.upper


def test_grouped_callback_candidate_is_replayed_before_head_merge() -> None:
  rule = SourceRule(
    text='answer(X):_max_first_clause <- evidence(X,Y)',
    name='callback',
    head_predicate='answer',
    head_terms=('X',),
    head_annotation='_max_first_clause',
    head_lower=0.0,
    head_upper=1.0,
    delay=0,
    clauses=(SourceClause('evidence', ('X', 'Y'), 0.0, 1.0),),
  )
  source = SourceProgram(
    rules=(rule,),
    facts=(),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(('_max_first_clause', _max_first_clause),),
  )
  rows = (
    TemporalIntervalRow('evidence', ('a', 'b'), 0, 0.6, 1.0),
    TemporalIntervalRow('answer', ('a',), 0, 0.8, 0.9),
  )

  graph = explain_demands(source, rows, (DemandSeed('answer', ('a',), 0),))

  answer = graph.atom(DemandSeed('answer', ('a',), 0))
  assert (answer.rules[0].candidate_lower, answer.rules[0].candidate_upper) == (0.6, 1.0)


def test_instrumented_candidate_keeps_cwa_version_after_final_narrowing() -> None:
  rule = SourceRule(
    text='safe(X) <- ~blocked(X)',
    name='safe-by-default',
    head_predicate='safe',
    head_terms=('X',),
    head_annotation=None,
    head_lower=1.0,
    head_upper=1.0,
    delay=0,
    clauses=(SourceClause('blocked', ('X',), 0.0, 0.0),),
  )
  source = SourceProgram(
    rules=(rule,),
    facts=(),
    graphml_path=None,
    closed_world_predicates=frozenset({'blocked'}),
    annotation_functions=(),
    node_domain=('a',),
  )
  rows = (
    TemporalIntervalRow('blocked', ('a',), 0, 1.0, 1.0),
    TemporalIntervalRow('safe', ('a',), 0, 1.0, 1.0),
  )
  candidate = RuleCandidateRow(
    rule_index=0,
    rule_name='safe-by-default',
    head_predicate='safe',
    head_arguments=('a',),
    head_time=0,
    rank=0,
    candidate_lower=1.0,
    candidate_upper=1.0,
    bindings=(('X', 'a'),),
    body_intervals=((0.0, 1.0),),
    closed_world_clause_positions=(0,),
  )

  graph = explain_demands(
    source,
    rows,
    (DemandSeed('safe', ('a',), 0),),
    rule_candidates=(candidate,),
  )

  explanation = graph.atom(graph.roots[0]).rules[0]
  assert explanation.grounding_body_intervals == (((0.0, 1.0),),)
  assert explanation.closed_world_clause_positions == (0,)
  assert explanation.witness_key[-2:] == (
    (((0.0, 1.0),),),
    (0,),
  )
  child = explanation.children[0]
  assert child == DemandSeed(
    'blocked',
    ('a',),
    0,
    observed_lower=0.0,
    observed_upper=1.0,
    evidence_mode='closed-world-default',
  )
  blocked = graph.atom(child)
  assert blocked is not None
  assert (blocked.materialized_lower, blocked.materialized_upper) == (0.0, 1.0)
  assert blocked.facts == ()
  assert blocked.rules == ()
  assert len(blocked.closed_world_assumptions) == 1


def test_crossed_cwa_witness_composes_repair_with_default_false() -> None:
  rule = SourceRule(
    text='bad(X) <- ~p(X)',
    name='bad-by-default',
    head_predicate='bad',
    head_terms=('X',),
    head_annotation=None,
    head_lower=1.0,
    head_upper=1.0,
    delay=0,
    clauses=(SourceClause('p', ('X',), 0.0, 0.0),),
  )
  source = SourceProgram(
    rules=(rule,),
    facts=(
      SourceFact('p(a):[0.8,1]', 'p-high', 'p', ('a',), 0.8, 1.0, 0, 0, True),
      SourceFact('p(a):[0,0.2]', 'p-low', 'p', ('a',), 0.0, 0.2, 0, 0, True),
    ),
    graphml_path=None,
    closed_world_predicates=frozenset({'p'}),
    annotation_functions=(),
    node_domain=('a',),
  )
  rows = (
    TemporalIntervalRow('p', ('a',), 0, 0.0, 1.0, True, 0.8, 0.2),
    TemporalIntervalRow('bad', ('a',), 0, 1.0, 1.0),
  )
  candidate = RuleCandidateRow(
    0,
    'bad-by-default',
    'bad',
    ('a',),
    0,
    0,
    1.0,
    1.0,
    (('X', 'a'),),
    ((0.8, 0.2),),
    (0,),
  )

  graph = explain_demands(
    source,
    rows,
    (DemandSeed('bad', ('a',), 0),),
    rule_candidates=(candidate,),
  )

  bad = graph.atom(graph.roots[0])
  cwa_read = bad.rules[0].children[0]
  assert cwa_read.evidence_mode == 'closed-world-default'
  cwa_atom = graph.atom(cwa_read)
  conflict = cwa_atom.closed_world_assumptions[0].source
  assert conflict == DemandSeed(
    'p',
    ('a',),
    0,
    observed_lower=0.8,
    observed_upper=0.2,
    evidence_mode='materialized',
  )
  conflict_atom = graph.atom(conflict)
  assert conflict_atom.inconsistent is True
  assert {fact.fact_name for fact in conflict_atom.facts} == {'p-high', 'p-low'}


def test_same_rule_and_grounding_at_two_interval_versions_are_distinct() -> None:
  source = SourceProgram(
    rules=(_rule('copy-mid', 'result', 'mid'),),
    facts=(),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(),
  )
  rows = (
    TemporalIntervalRow('mid', ('a',), 0, 0.0, 1.0, inconsistent=True),
    TemporalIntervalRow('result', ('a',), 0, 1.0, 1.0),
  )
  candidates = tuple(
    RuleCandidateRow(
      rule_index=0,
      rule_name='copy-mid',
      head_predicate='result',
      head_arguments=('a',),
      head_time=0,
      rank=0,
      candidate_lower=1.0,
      candidate_upper=1.0,
      bindings=(('X', 'a'),),
      body_intervals=(body_interval,),
    )
    for body_interval in ((0.6, 1.0), (1.0, 1.0))
  )

  graph = explain_demands(
    source,
    rows,
    (DemandSeed('result', ('a',), 0),),
    rule_candidates=candidates,
  )

  explanations = graph.atom(graph.roots[0]).rules
  assert len(explanations) == 2
  assert explanations[0].witness_key != explanations[1].witness_key
  assert {item.grounding_body_intervals for item in explanations} == {
    (((0.6, 1.0),),),
    (((1.0, 1.0),),),
  }
  assert len(graph.occurrences(DemandSeed('mid', ('a',), 0))) == 2


def test_historical_consistent_child_does_not_reopen_final_crossed_trace() -> None:
  source = SourceProgram(
    rules=(
      _rule('copy-consistent-mid', 'result', 'mid'),
      _rule('derive-mid-high', 'mid', 'seed'),
      _rule('derive-stage', 'stage', 'seed'),
      _rule('derive-trigger', 'trigger', 'stage'),
      _rule('cross-mid-later', 'mid', 'trigger', lower=0.0, upper=0.2),
    ),
    facts=(SourceFact('seed(a)', 'seed-a', 'seed', ('a',), 1, 1, 0, 0, True),),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(),
  )
  rows = (
    TemporalIntervalRow('seed', ('a',), 0, 1.0, 1.0),
    TemporalIntervalRow('stage', ('a',), 0, 1.0, 1.0),
    TemporalIntervalRow('trigger', ('a',), 0, 1.0, 1.0),
    TemporalIntervalRow(
      'mid',
      ('a',),
      0,
      0.0,
      1.0,
      inconsistent=True,
      raw_lower=1.0,
      raw_upper=0.2,
    ),
    TemporalIntervalRow('result', ('a',), 0, 1.0, 1.0),
  )
  candidates = (
    RuleCandidateRow(
      0,
      'copy-consistent-mid',
      'result',
      ('a',),
      0,
      0,
      1.0,
      1.0,
      (('X', 'a'),),
      ((1.0, 1.0),),
    ),
    RuleCandidateRow(
      1,
      'derive-mid-high',
      'mid',
      ('a',),
      0,
      1,
      1.0,
      1.0,
      (('X', 'a'),),
      ((1.0, 1.0),),
    ),
    RuleCandidateRow(
      4,
      'cross-mid-later',
      'mid',
      ('a',),
      0,
      4,
      0.0,
      0.2,
      (('X', 'a'),),
      ((1.0, 1.0),),
    ),
  )

  graph = explain_demands(
    source,
    rows,
    (DemandSeed('result', ('a',), 0),),
    rule_candidates=candidates,
  )

  result = graph.atom(graph.roots[0])
  assert result is not None
  child = result.rules[0].children[0]
  assert (child.observed_lower, child.observed_upper) == (1.0, 1.0)
  mid = graph.atom(child)
  assert mid is not None
  assert (mid.materialized_lower, mid.materialized_upper, mid.inconsistent) == (
    1.0,
    1.0,
    False,
  )
  assert [rule.rule_index for rule in mid.rules] == [1]
  assert graph.atom(DemandSeed('trigger', ('a',), 0)) is None


def test_label_reset_is_an_on_demand_temporal_witness() -> None:
  source = SourceProgram(
    rules=(),
    facts=(SourceFact('', 'p-now', 'p', ('a',), 1.0, 1.0, 0, 0, False),),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(),
  )
  rows = (
    TemporalIntervalRow('p', ('a',), 0, 1.0, 1.0),
    TemporalIntervalRow('p', ('a',), 1, 0.0, 1.0),
  )

  graph = explain_demands(source, rows, (DemandSeed('p', ('a',), 1),))

  current = graph.atom(DemandSeed('p', ('a',), 1))
  assert current is not None
  assert [(edge.kind, edge.source.time) for edge in current.transitions] == [('label-reset', 0)]
  previous = graph.atom(current.transitions[0].source)
  assert previous is not None
  assert [fact.fact_name for fact in previous.facts] == ['p-now']


def test_persistent_frame_is_an_on_demand_temporal_witness() -> None:
  source = SourceProgram(
    rules=(),
    facts=(SourceFact('', 'p-now', 'p', ('a',), 0.6, 0.8, 0, 0, False),),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(),
    settings=(('persistent', True),),
  )
  rows = (
    TemporalIntervalRow('p', ('a',), 0, 0.6, 0.8),
    TemporalIntervalRow('p', ('a',), 1, 0.6, 0.8),
  )

  graph = explain_demands(source, rows, (DemandSeed('p', ('a',), 1),))

  current = graph.atom(DemandSeed('p', ('a',), 1))
  assert current is not None
  assert [(edge.kind, edge.source.time) for edge in current.transitions] == [
    ('persistent-frame', 0)
  ]


def test_conflict_freeze_is_one_origin_plus_a_temporal_witness() -> None:
  source = SourceProgram(
    rules=(),
    facts=(
      SourceFact('', 'left', 'p', ('a',), 0.8, 1.0, 0, 0, False),
      SourceFact('', 'right', 'p', ('a',), 0.0, 0.2, 0, 0, False),
    ),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(),
  )
  rows = (
    TemporalIntervalRow(
      'p',
      ('a',),
      0,
      0.0,
      1.0,
      inconsistent=True,
      raw_lower=1.0,
      raw_upper=0.0,
      frozen=True,
    ),
    TemporalIntervalRow(
      'p',
      ('a',),
      1,
      0.0,
      1.0,
      raw_lower=1.0,
      raw_upper=0.0,
      frozen=True,
    ),
  )

  graph = explain_demands(source, rows, (DemandSeed('p', ('a',), 1),))

  current = graph.atom(DemandSeed('p', ('a',), 1))
  assert current is not None
  assert current.inconsistent is False
  assert [(edge.kind, edge.source.time) for edge in current.transitions] == [('conflict-freeze', 0)]
  origin = graph.atom(current.transitions[0].source)
  assert origin is not None
  assert origin.inconsistent is True
  assert {fact.fact_name for fact in origin.facts} == {'left', 'right'}


def test_explicit_unknown_cwa_read_composes_materialized_fact_proof() -> None:
  rule = SourceRule(
    text='bad(X) <- p(X):[0,0]',
    name='default',
    head_predicate='bad',
    head_terms=('X',),
    head_annotation=None,
    head_lower=1.0,
    head_upper=1.0,
    delay=0,
    clauses=(SourceClause('p', ('X',), 0.0, 0.0),),
  )
  source = SourceProgram(
    rules=(rule,),
    facts=(SourceFact('', 'unknown', 'p', ('a',), 0.0, 1.0, 0, 0, False),),
    graphml_path=None,
    closed_world_predicates=frozenset({'p'}),
    annotation_functions=(),
    node_domain=('a',),
  )
  rows = (
    TemporalIntervalRow('p', ('a',), 0, 0.0, 1.0),
    TemporalIntervalRow('bad', ('a',), 0, 1.0, 1.0),
  )

  graph = explain_demands(source, rows, (DemandSeed('bad', ('a',), 0),))

  default_occurrence = next(
    atom
    for atom in graph.atoms
    if atom.demand.predicate == 'p' and atom.demand.evidence_mode == 'closed-world-default'
  )
  assumption = default_occurrence.closed_world_assumptions[0]
  assert assumption.source is not None
  materialized = graph.atom(assumption.source)
  assert materialized is not None
  assert [fact.fact_name for fact in materialized.facts] == ['unknown']
