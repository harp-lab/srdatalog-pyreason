from srdatalog.pyreason.demand import (
  ConstantIntervalTransform,
  DemandSeed,
  GroupedCallbackTransform,
  ground_one_hop_witnesses,
  pyreason_clause_components,
  rewrite_for_demands,
)
from srdatalog.pyreason.model import (
  SourceClause,
  SourceFact,
  SourceProgram,
  SourceRule,
  TemporalIntervalRow,
)


def clause(
  predicate: str,
  *terms: str,
  lower: float = 0.0,
  upper: float = 1.0,
) -> SourceClause:
  return SourceClause(predicate, terms, lower, upper)


def rule(
  name: str,
  head: str,
  head_terms: tuple[str, ...],
  *body: SourceClause,
  delay: int = 0,
  annotation: str | None = None,
  head_interval: tuple[float, float] = (1.0, 1.0),
) -> SourceRule:
  return SourceRule(
    text=f'{head}{head_terms} <- ...',
    name=name,
    head_predicate=head,
    head_terms=head_terms,
    head_annotation=annotation,
    head_lower=head_interval[0],
    head_upper=head_interval[1],
    delay=delay,
    clauses=body,
    weights=tuple(1.0 for _ in body),
  )


def program(
  *rules: SourceRule,
  facts: tuple[SourceFact, ...] = (),
) -> SourceProgram:
  return SourceProgram(
    rules=rules,
    facts=facts,
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(),
  )


def test_demand_prunes_rules_and_unreachable_predicates() -> None:
  source = program(
    rule('wanted', 'answer', ('X',), clause('input', 'X')),
    rule('unrelated', 'other', ('X',), clause('noise', 'X')),
  )

  rewrite = rewrite_for_demands(source, (DemandSeed('answer', ('a',), 0),))

  assert [witness.identity.rule_name for witness in rewrite.rule_witnesses] == ['wanted']
  assert [(demand.predicate, demand.arguments) for demand in rewrite.demands] == [
    ('answer', ('a',)),
    ('input', ('a',)),
  ]


def test_two_rules_for_same_fact_keep_distinct_witnesses_and_transforms() -> None:
  source = program(
    rule(
      'low-rule',
      'answer',
      ('X',),
      clause('left', 'X'),
      head_interval=(0.2, 0.4),
    ),
    rule(
      'callback-rule',
      'answer',
      ('X',),
      clause('right', 'X'),
      annotation='choose_bound',
    ),
  )

  rewrite = rewrite_for_demands(source, (DemandSeed('answer', ('a',), 0),))

  identities = [witness.identity for witness in rewrite.rule_witnesses]
  assert [(identity.rule_index, identity.rule_name) for identity in identities] == [
    (0, 'low-rule'),
    (1, 'callback-rule'),
  ]
  assert identities[0].demanded_head == identities[1].demanded_head
  assert isinstance(rewrite.rule_witnesses[0].interval_transform, ConstantIntervalTransform)
  callback = rewrite.rule_witnesses[1].interval_transform
  assert isinstance(callback, GroupedCallbackTransform)
  assert callback.callback.annotation_name == 'choose_bound'


def test_zero_delay_recursion_terminates_by_demand_key() -> None:
  source = program(
    rule('recursive', 'path', ('X',), clause('path', 'X')),
  )

  rewrite = rewrite_for_demands(source, (DemandSeed('path', ('a',), 3),))

  assert len(rewrite.demands) == 1
  assert len(rewrite.rule_witnesses) == 1
  assert rewrite.rule_witnesses[0].body[0].demand == rewrite.demands[0]


def test_delay_propagates_a_body_demand_backward_in_time() -> None:
  source = program(
    rule('tomorrow', 'future', ('X',), clause('present', 'X'), delay=2),
  )

  rewrite = rewrite_for_demands(source, (DemandSeed('future', ('a',), 5),))

  witness = rewrite.rule_witnesses[0]
  assert witness.delay == 2
  assert witness.body[0].time == 3
  assert rewrite.demands[-1].time == 3


def test_matching_facts_are_named_leaves_and_stop_the_branch() -> None:
  facts = (
    SourceFact('seed(a)', 'seed-a', 'seed', ('a',), 0.6, 0.8, 0, 0, True),
    SourceFact('seed(b)', 'seed-b', 'seed', ('b',), 1.0, 1.0, 0, 10, False),
  )
  source = program(facts=facts)

  rewrite = rewrite_for_demands(source, (DemandSeed('seed', ('a',), 7),))

  assert len(rewrite.fact_leaves) == 1
  leaf = rewrite.fact_leaves[0]
  assert leaf.identity.fact_index == 0
  assert leaf.identity.fact_name == 'seed-a'
  assert leaf.identity.time == 7
  assert (leaf.lower, leaf.upper) == (0.6, 0.8)
  assert rewrite.rule_witnesses == ()


def test_grounded_one_hop_witness_builds_pyreason_clause_components() -> None:
  source = program(
    rule(
      'join',
      'answer',
      ('X',),
      clause('nodeLabel', 'X', lower=0.5, upper=1.0),
      clause('edgeLabel', 'X', 'Y', lower=0.25, upper=1.0),
    ),
  )
  rewrite = rewrite_for_demands(source, (DemandSeed('answer', ('a',), 4),))
  rows = (
    TemporalIntervalRow('edgeLabel', ('a', 'b'), 4, 0.5, 0.75),
    TemporalIntervalRow('nodeLabel', ('a',), 4, 0.7, 0.9),
    TemporalIntervalRow('edgeLabel', ('a', 'ignored'), 3, 1.0, 1.0),
  )

  grounded = ground_one_hop_witnesses(rewrite.rule_witnesses[0], rows)
  assert [dict(item.bindings) for item in grounded] == [{'X': 'a', 'Y': 'b'}]

  components = pyreason_clause_components(grounded[0], rows)
  assert [(component.position, component.predicate) for component in components] == [
    (1, 'nodeLabel'),
    (2, 'edgeLabel'),
  ]
  assert components[0].as_trace_cell() == ['a']
  assert components[1].as_trace_cell() == [('a', 'b')]
