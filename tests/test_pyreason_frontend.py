from pathlib import Path

import pytest

from srdatalog import float32_to_u32
from srdatalog.dsl import Atom, Let
from srdatalog.pyreason import (
  DemandSeed,
  NativePlan,
  SourceClause,
  SourceFact,
  SourceProgram,
  SourceRule,
  UnsupportedAnnotationSemantics,
  compile_source,
)


def annotation(annotations, weights):
  return annotations[0][0].lower, annotations[0][0].upper


def six_argument_annotation(
  annotations,
  weights,
  qualified_nodes,
  qualified_edges,
  clause_labels,
  clause_variables,
):
  best_lower = 0.0
  best_upper = 0.0
  for row in range(len(annotations[0])):
    candidate_lower = annotations[0][row].lower
    candidate_upper = annotations[0][row].upper
    if candidate_lower > best_lower:
      best_lower = candidate_lower
      best_upper = candidate_upper
  return best_lower, best_upper


def first_candidate_annotation(
  annotations,
  weights,
  qualified_nodes,
  qualified_edges,
  clause_labels,
  clause_variables,
):
  best_lower = 0.0
  best_upper = 0.0
  found_any = False
  for row in range(len(annotations[0])):
    candidate_lower = annotations[0][row].lower
    candidate_upper = annotations[0][row].upper
    if not found_any or candidate_lower > best_lower:
      best_lower = candidate_lower
      best_upper = candidate_upper
      found_any = True
  if not found_any:
    return 0.0, 1.0
  return best_lower, best_upper


def existential_payload_annotation(
  annotations,
  weights,
  qualified_nodes,
  qualified_edges,
  clause_labels,
  clause_variables,
):
  best_lower = 0.0
  best_upper = 0.0
  for driver_row in range(len(annotations[0])):
    for context_row in range(len(annotations[2])):
      candidate_lower = min(
        annotations[0][driver_row].lower,
        annotations[2][context_row].lower,
      )
      candidate_upper = min(
        annotations[0][driver_row].upper,
        annotations[2][context_row].upper,
      )
      if candidate_lower > best_lower:
        best_lower = candidate_lower
        best_upper = candidate_upper
  return best_lower, best_upper


def crossed_winner_annotation(
  annotations,
  weights,
  qualified_nodes,
  qualified_edges,
  clause_labels,
  clause_variables,
):
  best_lower = 0.0
  best_upper = 0.0
  for driver_row in range(len(annotations[0])):
    for context_row in range(len(annotations[1])):
      candidate_lower = max(
        annotations[0][driver_row].lower,
        annotations[1][context_row].lower,
      )
      candidate_upper = min(
        annotations[0][driver_row].upper,
        annotations[1][context_row].upper,
      )
      if candidate_lower > best_lower:
        best_lower = candidate_lower
        best_upper = candidate_upper
  if best_lower > best_upper:
    return 0.0, 1.0
  return best_lower, best_upper


def guarded_selection_annotation(
  annotations,
  weights,
  qualified_nodes,
  qualified_edges,
  clause_labels,
  clause_variables,
):
  best_lower = 0.0
  best_upper = 0.0
  for row in range(len(annotations[0])):
    candidate_lower = annotations[0][row].lower
    candidate_upper = annotations[0][row].upper
    if candidate_lower > best_lower and candidate_upper < 0.5:
      best_lower = candidate_lower
      best_upper = candidate_upper
  return best_lower, best_upper


def selection_else_annotation(
  annotations,
  weights,
  qualified_nodes,
  qualified_edges,
  clause_labels,
  clause_variables,
):
  best_lower = 0.0
  best_upper = 0.0
  for row in range(len(annotations[0])):
    candidate_lower = annotations[0][row].lower
    candidate_upper = annotations[0][row].upper
    if candidate_lower > best_lower:
      best_lower = candidate_lower
      best_upper = candidate_upper
    else:
      best_upper = candidate_upper
  return best_lower, best_upper


def repeated_clause_loop_annotation(
  annotations,
  weights,
  qualified_nodes,
  qualified_edges,
  clause_labels,
  clause_variables,
):
  best_lower = 0.0
  best_upper = 0.0
  for outer in range(len(annotations[0])):
    for inner in range(len(annotations[0])):
      candidate_lower = annotations[0][outer].lower
      candidate_upper = annotations[0][inner].upper
      if candidate_lower > best_lower:
        best_lower = candidate_lower
        best_upper = candidate_upper
  return best_lower, best_upper


def unrelated_fallback_annotation(
  annotations,
  weights,
  qualified_nodes,
  qualified_edges,
  clause_labels,
  clause_variables,
):
  best_lower = 0.0
  best_upper = 0.0
  for row in range(len(annotations[0])):
    candidate_lower = annotations[0][row].lower
    candidate_upper = annotations[0][row].upper
    if candidate_lower > best_lower:
      best_lower = candidate_lower
      best_upper = candidate_upper
  if best_upper > best_lower:
    return 0.0, 1.0
  return best_lower, best_upper


def boolean_crossed_fallback_annotation(
  annotations,
  weights,
  qualified_nodes,
  qualified_edges,
  clause_labels,
  clause_variables,
):
  best_lower = 0.0
  best_upper = 0.0
  for row in range(len(annotations[0])):
    candidate_lower = annotations[0][row].lower
    candidate_upper = annotations[0][row].upper
    if candidate_lower > best_lower:
      best_lower = candidate_lower
      best_upper = candidate_upper
  always = True
  if always or best_lower > best_upper:
    return 0.0, 1.0
  return best_lower, best_upper


def negated_crossed_fallback_annotation(
  annotations,
  weights,
  qualified_nodes,
  qualified_edges,
  clause_labels,
  clause_variables,
):
  best_lower = 0.0
  best_upper = 0.0
  for row in range(len(annotations[0])):
    candidate_lower = annotations[0][row].lower
    candidate_upper = annotations[0][row].upper
    if candidate_lower > best_lower:
      best_lower = candidate_lower
      best_upper = candidate_upper
  if not (best_lower > best_upper):
    return 0.0, 1.0
  return best_lower, best_upper


def boolean_lookup_annotation(
  annotations,
  weights,
  qualified_nodes,
  qualified_edges,
  clause_labels,
  clause_variables,
):
  best_lower = 0.0
  best_upper = 0.0
  for driver in range(len(annotations[0])):
    lookup_lower = -1.0
    lookup_upper = -1.0
    never = False
    for row in range(len(annotations[1])):
      if never and qualified_nodes[1][row] == qualified_nodes[0][driver]:
        lookup_lower = annotations[1][row].lower
        lookup_upper = annotations[1][row].upper
        break
    candidate_lower = min(annotations[0][driver].lower, lookup_lower)
    candidate_upper = min(annotations[0][driver].upper, lookup_upper)
    if candidate_lower > best_lower:
      best_lower = candidate_lower
      best_upper = candidate_upper
  return best_lower, best_upper


def nonterminal_selection_annotation(
  annotations,
  weights,
  qualified_nodes,
  qualified_edges,
  clause_labels,
  clause_variables,
):
  best_lower = 0.0
  best_upper = 0.0
  for row in range(len(annotations[0])):
    candidate_lower = annotations[0][row].lower
    candidate_upper = annotations[0][row].upper
    if candidate_lower > best_lower:
      best_lower = candidate_lower
      best_upper = candidate_upper
    best_upper = candidate_upper
  return best_lower, best_upper


def nonterminal_lookup_annotation(
  annotations,
  weights,
  qualified_nodes,
  qualified_edges,
  clause_labels,
  clause_variables,
):
  best_lower = 0.0
  best_upper = 0.0
  for driver in range(len(annotations[0])):
    lookup_lower = -1.0
    lookup_upper = -1.0
    for row in range(len(annotations[1])):
      if qualified_nodes[1][row] == qualified_nodes[0][driver]:
        lookup_lower = annotations[1][row].lower
        lookup_upper = annotations[1][row].upper
        break
      lookup_upper = annotations[1][row].upper
    candidate_lower = min(annotations[0][driver].lower, lookup_lower)
    candidate_upper = min(annotations[0][driver].upper, lookup_upper)
    if candidate_lower > best_lower:
      best_lower = candidate_lower
      best_upper = candidate_upper
  return best_lower, best_upper


def loop_else_annotation(
  annotations,
  weights,
  qualified_nodes,
  qualified_edges,
  clause_labels,
  clause_variables,
):
  best_lower = 0.0
  best_upper = 0.0
  for row in range(len(annotations[0])):
    candidate_lower = annotations[0][row].lower
    candidate_upper = annotations[0][row].upper
    if candidate_lower > best_lower:
      best_lower = candidate_lower
      best_upper = candidate_upper
  else:
    best_upper = 1.0
  return best_lower, best_upper


def nested_nonterminal_selection_annotation(
  annotations,
  weights,
  qualified_nodes,
  qualified_edges,
  clause_labels,
  clause_variables,
):
  best_lower = 0.0
  best_upper = 0.0
  enabled = True
  for row in range(len(annotations[0])):
    candidate_lower = annotations[0][row].lower
    candidate_upper = annotations[0][row].upper
    if enabled:
      if candidate_lower > best_lower:
        best_lower = candidate_lower
        best_upper = candidate_upper
    best_upper = candidate_upper
  return best_lower, best_upper


def sequential_loop_escape_annotation(
  annotations,
  weights,
  qualified_nodes,
  qualified_edges,
  clause_labels,
  clause_variables,
):
  saved_lower = 0.0
  for left_row in range(len(annotations[0])):
    saved_lower = annotations[0][left_row].lower

  best_lower = 0.0
  best_upper = 0.0
  for right_row in range(len(annotations[1])):
    candidate_lower = min(saved_lower, annotations[1][right_row].lower)
    candidate_upper = annotations[1][right_row].upper
    if candidate_lower > best_lower:
      best_lower = candidate_lower
      best_upper = candidate_upper
  return best_lower, best_upper


def loop_carried_candidate_annotation(
  annotations,
  weights,
  qualified_nodes,
  qualified_edges,
  clause_labels,
  clause_variables,
):
  best_lower = 0.0
  best_upper = 0.0
  for row in range(len(annotations[0])):
    candidate_lower = min(best_lower, annotations[0][row].lower)
    candidate_upper = annotations[0][row].upper
    if candidate_lower > best_lower:
      best_lower = candidate_lower
      best_upper = candidate_upper
  return best_lower, best_upper


def stale_lookup_sentinel_annotation(
  annotations,
  weights,
  qualified_nodes,
  qualified_edges,
  clause_labels,
  clause_variables,
):
  best_lower = 0.0
  best_upper = 0.0
  lookup_lower = -1.0
  lookup_upper = -1.0
  for driver in range(len(annotations[0])):
    for row in range(len(annotations[1])):
      if qualified_nodes[1][row] == qualified_nodes[0][driver]:
        lookup_lower = annotations[1][row].lower
        lookup_upper = annotations[1][row].upper
        break
    candidate_lower = min(annotations[0][driver].lower, lookup_lower)
    candidate_upper = min(annotations[0][driver].upper, lookup_upper)
    if candidate_lower > best_lower:
      best_lower = candidate_lower
      best_upper = candidate_upper
  return best_lower, best_upper


def self_equal_lookup_annotation(
  annotations,
  weights,
  qualified_nodes,
  qualified_edges,
  clause_labels,
  clause_variables,
):
  best_lower = 0.0
  best_upper = 0.0
  for driver in range(len(annotations[0])):
    lookup_lower = -1.0
    lookup_upper = -1.0
    for row in range(len(annotations[1])):
      if qualified_nodes[1][row] == qualified_nodes[1][row]:
        lookup_lower = annotations[1][row].lower
        lookup_upper = annotations[1][row].upper
        break
    candidate_lower = min(annotations[0][driver].lower, lookup_lower)
    candidate_upper = min(annotations[0][driver].upper, lookup_upper)
    if candidate_lower > best_lower:
      best_lower = candidate_lower
      best_upper = candidate_upper
  return best_lower, best_upper


def test_source_program_preserves_plain_registered_callable() -> None:
  rule = SourceRule(
    text='output(x):annotation <- input(x)',
    name='copy-bound',
    head_predicate='output',
    head_terms=('x',),
    head_annotation='annotation',
    head_lower=0.0,
    head_upper=1.0,
    delay=0,
    clauses=(),
  )
  source = SourceProgram(
    rules=(rule,),
    facts=(),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(('annotation', annotation),),
  )

  assert dict(source.annotation_functions)['annotation'] is annotation
  assert not hasattr(annotation, '__srdatalog_annotation_semantics__')
  assert source.node_domain == ()
  assert source.edge_domain == ()
  assert source.inconsistent_predicates == ()


def test_source_program_preserves_neutral_domains_and_inconsistency_pairs() -> None:
  source = SourceProgram(
    rules=(),
    facts=(),
    graphml_path=None,
    closed_world_predicates=frozenset({'missing'}),
    annotation_functions=(),
    node_domain=('a', 'b'),
    edge_domain=(('a', 'b'),),
    inconsistent_predicates=(('present', 'absent'),),
  )

  assert source.node_domain == ('a', 'b')
  assert source.edge_domain == (('a', 'b'),)
  assert source.inconsistent_predicates == (('present', 'absent'),)


def test_target_package_owns_compile_api(tmp_path: Path) -> None:
  clause = SourceClause(
    predicate='input',
    terms=('x',),
    lower=0.0,
    upper=1.0,
  )
  rule = SourceRule(
    text='output(x):six_argument_annotation <- input(x)',
    name='copy-bound',
    head_predicate='output',
    head_terms=('x',),
    head_annotation='six_argument_annotation',
    head_lower=0.0,
    head_upper=1.0,
    delay=0,
    clauses=(clause,),
  )
  fact = SourceFact(
    text='input(a):[0.5,0.75]',
    name='input-a',
    predicate='input',
    arguments=('a',),
    lower=0.5,
    upper=0.75,
    start_time=0,
    end_time=0,
    static=True,
  )
  source = SourceProgram(
    rules=(rule,),
    facts=(fact,),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(('six_argument_annotation', six_argument_annotation),),
  )

  assert compile_source.__module__ == 'srdatalog.pyreason.compiler'
  assert issubclass(UnsupportedAnnotationSemantics, NotImplementedError)
  plan = compile_source(source, timesteps=0, output_root=tmp_path)
  assert isinstance(plan, NativePlan)
  assert plan.rewriter == 'pyreason-srdatalog-core-v9'
  assert plan.callback_read_predicates == ('input',)
  assert {output.predicate for output in plan.outputs} == {'input', 'output'}
  assert [rule.name for rule in plan.program.rules] == [
    'PyReasonLiveThroughHorizon',
    'LoadSeedPyReasonRelation0',
    'PyReasonRoundBridge0',
    'PyReasonRule0_copy_bound__aggregate_candidate',
    'PyReasonRule0_copy_bound__aggregate_select',
    'PyReasonRule0_copy_bound__aggregate_initializer',
    'PyReasonRule0_copy_bound__aggregate_effective',
    'PyReasonRule0_copy_bound',
  ]
  callback_rule = next(
    rule
    for rule in plan.program.rules
    if rule.name == 'PyReasonRule0_copy_bound__aggregate_candidate'
  )
  atoms = [clause for clause in callback_rule.body if isinstance(clause, Atom)]
  assert [atom.rel for atom in atoms[:2]] == [
    'PyReasonAdmissionRank0',
    'PyReasonRelation0',
  ]
  # Rank comes from the source declaration, but callback endpoints come from
  # the interval-lattice world after same-key facts have been merged.
  assert atoms[0].args[-2:] != atoms[1].args[-2:]

  assert callback_rule.head.rel == 'PyReasonCallback0ProposalCandidate'
  assert callback_rule.head.relation is not None
  assert callback_rule.head.relation.value_spec is None
  selection_rule = next(
    rule for rule in plan.program.rules if rule.name == 'PyReasonRule0_copy_bound__aggregate_select'
  )
  assert selection_rule.head.rel == 'PyReasonCallback0SelectedProposal'
  assert selection_rule.head.relation is not None
  assert selection_rule.head.relation.value_spec is not None
  assert selection_rule.head.relation.value_spec.join.value == 'max-lower-select'
  selection_atoms = [clause for clause in selection_rule.body if isinstance(clause, Atom)]
  assert [atom.rel for atom in selection_atoms] == ['PyReasonCallback0ProposalCandidate']

  # No variables remain outside the head and driver key, so one admission rank
  # already determines a unique logical grounding and needs no runtime guard.
  assert plan.callback_fd_relations == ()

  initializer = next(
    rule
    for rule in plan.program.rules
    if rule.name == 'PyReasonRule0_copy_bound__aggregate_initializer'
  )
  # A plain strict maximum starts from [0,0].  Rank zero is reserved for
  # that initializer, while source admissions start at one, so [0,0] wins
  # an exact lower-bound tie exactly as the Python ``>`` test does.
  assert initializer.head.args[-3].const_value == 0
  assert initializer.head.args[-2].const_value == float32_to_u32(0.0)
  assert initializer.head.args[-1].const_value == float32_to_u32(0.0)


def test_callback_rank_functionally_determines_vulreasoner_shaped_payload(
  tmp_path: Path,
) -> None:
  rule = SourceRule(
    text=('output(H):six_argument_annotation <- driver(L,R), leftLabel(H,L), rightLabel(H,R)'),
    name='functional-callback-group',
    head_predicate='output',
    head_terms=('H',),
    head_annotation='six_argument_annotation',
    head_lower=0.0,
    head_upper=1.0,
    delay=0,
    clauses=(
      SourceClause('driver', ('L', 'R'), 0.1, 1.0),
      SourceClause('leftLabel', ('H', 'L'), 0.1, 1.0),
      SourceClause('rightLabel', ('H', 'R'), 0.1, 1.0),
    ),
  )
  source = SourceProgram(
    rules=(rule,),
    facts=(
      SourceFact('driver(l,r)', 'driver', 'driver', ('l', 'r'), 0.5, 0.8, 0, 0, True),
      SourceFact(
        'leftLabel(h,l)',
        'left-label',
        'leftLabel',
        ('h', 'l'),
        0.6,
        0.9,
        0,
        0,
        True,
      ),
      SourceFact(
        'rightLabel(h,r)',
        'right-label',
        'rightLabel',
        ('h', 'r'),
        0.7,
        1.0,
        0,
        0,
        True,
      ),
    ),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(('six_argument_annotation', six_argument_annotation),),
  )

  plan = compile_source(source, timesteps=0, output_root=tmp_path)

  assert any(
    rule.name == 'PyReasonRule0_functional_callback_group__aggregate_candidate'
    for rule in plan.program.rules
  )


def test_callback_extra_existential_payload_gets_runtime_fd_guard(
  tmp_path: Path,
) -> None:
  rule = SourceRule(
    text=('output(H):existential_payload_annotation <- driver(L,R), leftLabel(H,L), context(X)'),
    name='non-functional-callback-group',
    head_predicate='output',
    head_terms=('H',),
    head_annotation='existential_payload_annotation',
    head_lower=0.0,
    head_upper=1.0,
    delay=0,
    clauses=(
      SourceClause('driver', ('L', 'R'), 0.1, 1.0),
      SourceClause('leftLabel', ('H', 'L'), 0.1, 1.0),
      SourceClause('context', ('X',), 0.1, 1.0),
    ),
  )
  source = SourceProgram(
    rules=(rule,),
    facts=(
      SourceFact('driver(l,r)', 'driver', 'driver', ('l', 'r'), 0.5, 0.8, 0, 0, True),
      SourceFact(
        'leftLabel(h,l)',
        'left-label',
        'leftLabel',
        ('h', 'l'),
        0.6,
        0.9,
        0,
        0,
        True,
      ),
      SourceFact('context(x)', 'context-x', 'context', ('x',), 0.7, 1.0, 0, 0, True),
      SourceFact('context(y)', 'context-y', 'context', ('y',), 0.4, 0.9, 0, 0, True),
    ),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(('existential_payload_annotation', existential_payload_annotation),),
  )

  plan = compile_source(source, timesteps=0, output_root=tmp_path)
  violation_rule = next(
    rule
    for rule in plan.program.rules
    if rule.name == 'PyReasonRule0_non_functional_callback_group__aggregate_rank_fd_violation'
  )
  candidate_atoms = [clause for clause in violation_rule.body if isinstance(clause, Atom)]
  assert len(candidate_atoms) == 2
  assert candidate_atoms[0].rel == candidate_atoms[1].rel
  # The self-join fixes the full aggregate group and the driver's admission
  # rank, then compares both payloads and the retained logical X binding. It
  # rejects divergent X groundings without confusing a later interval version
  # of the same grounding with a physical row identity.
  assert candidate_atoms[0].args[:4] == candidate_atoms[1].args[:4]
  assert candidate_atoms[0].args[4:6] != candidate_atoms[1].args[4:6]
  assert candidate_atoms[0].args[-1] != candidate_atoms[1].args[-1]
  assert plan.callback_fd_relations == ('PyReasonCallback0RankFunctionalViolation',)


def test_first_candidate_fallback_loses_equal_lower_tie_to_real_admission(
  tmp_path: Path,
) -> None:
  clause = SourceClause('input', ('x',), 0.0, 1.0)
  source = SourceProgram(
    rules=(
      SourceRule(
        text='output(x):first_candidate_annotation <- input(x)',
        name='first-candidate',
        head_predicate='output',
        head_terms=('x',),
        head_annotation='first_candidate_annotation',
        head_lower=0.0,
        head_upper=1.0,
        delay=0,
        clauses=(clause,),
      ),
    ),
    facts=(SourceFact('input(a)', 'input-a', 'input', ('a',), 0, 1, 0, 0, True),),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(('first_candidate_annotation', first_candidate_annotation),),
  )

  plan = compile_source(source, timesteps=0, output_root=tmp_path)
  initializer = next(
    rule
    for rule in plan.program.rules
    if rule.name == 'PyReasonRule0_first_candidate__aggregate_initializer'
  )
  assert initializer.head.args[-3].const_value == 0xFFFFFFFF
  assert initializer.head.args[-2].const_value == float32_to_u32(0.0)
  assert initializer.head.args[-1].const_value == float32_to_u32(1.0)

  input_output = next(output for output in plan.outputs if output.predicate == 'input')
  input_index = int(input_output.relation.removeprefix('PyReasonRelation'))
  admission_row = (plan.data_dir / f'admission_rank_{input_index}.csv').read_text().splitlines()
  assert admission_row[0].split(',')[-1] == '1'


def test_crossed_callback_fallback_happens_after_winner_selection(
  tmp_path: Path,
) -> None:
  source = SourceProgram(
    rules=(
      SourceRule(
        text=('output(H):crossed_winner_annotation <- input(Y), context(H)'),
        name='crossed-winner',
        head_predicate='output',
        head_terms=('H',),
        head_annotation='crossed_winner_annotation',
        head_lower=0.0,
        head_upper=1.0,
        delay=1,
        clauses=(
          SourceClause('input', ('Y',), 0.0, 1.0),
          SourceClause('context', ('H',), 0.0, 1.0),
        ),
      ),
    ),
    facts=(
      SourceFact('input(a)', 'input-a', 'input', ('a',), 0.8, 1.0, 0, 0, True),
      SourceFact('input(b)', 'input-b', 'input', ('b',), 0.6, 0.7, 0, 0, True),
      SourceFact('context(h)', 'context-h', 'context', ('h',), 0.0, 0.7, 0, 0, True),
    ),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(('crossed_winner_annotation', crossed_winner_annotation),),
  )

  plan = compile_source(source, timesteps=1, output_root=tmp_path)
  names = {rule.name for rule in plan.program.rules}
  assert 'PyReasonRule0_crossed_winner__aggregate_effective' in names
  assert 'PyReasonRule0_crossed_winner__aggregate_crossed_fallback' in names

  fallback = next(
    rule
    for rule in plan.program.rules
    if rule.name == 'PyReasonRule0_crossed_winner__aggregate_crossed_fallback'
  )
  assert fallback.head.rel == 'PyReasonCallback0EffectiveProposal'
  assert fallback.head.args[-2].const_value == float32_to_u32(0.0)
  assert fallback.head.args[-1].const_value == float32_to_u32(1.0)

  materialization = next(
    rule for rule in plan.program.rules if rule.name == 'PyReasonRule0_crossed_winner'
  )
  materialization_atoms = [clause for clause in materialization.body if isinstance(clause, Atom)]
  assert [atom.rel for atom in materialization_atoms] == ['PyReasonCallback0EffectiveProposal']
  activation = next(
    rule for rule in plan.program.rules if rule.name == 'PyReasonRule0_crossed_winner__update_event'
  )
  assert (
    next(clause for clause in activation.body if isinstance(clause, Atom)).rel
    == 'PyReasonCallback0EffectiveProposal'
  )


def test_callback_rejects_nonmonotone_endpoint_orientation(
  tmp_path: Path,
) -> None:
  def upper_as_lower(
    annotations,
    weights,
    qualified_nodes,
    qualified_edges,
    clause_labels,
    clause_variables,
  ):
    best_lower = 0.0
    best_upper = 0.0
    for row in range(len(annotations[0])):
      candidate_lower = annotations[0][row].upper
      candidate_upper = annotations[0][row].lower
      if candidate_lower > best_lower:
        best_lower = candidate_lower
        best_upper = candidate_upper
    return best_lower, best_upper

  source = SourceProgram(
    rules=(
      SourceRule(
        text='output(X):upper_as_lower <- input(X)',
        name='nonmonotone-endpoints',
        head_predicate='output',
        head_terms=('X',),
        head_annotation='upper_as_lower',
        head_lower=0.0,
        head_upper=1.0,
        delay=0,
        clauses=(SourceClause('input', ('X',), 0.0, 1.0),),
      ),
    ),
    facts=(SourceFact('input(a)', 'input-a', 'input', ('a',), 0.8, 1.0, 0, 0, True),),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(('upper_as_lower', upper_as_lower),),
  )

  with pytest.raises(
    UnsupportedAnnotationSemantics,
    match='lower must depend only on source lower endpoints',
  ):
    compile_source(source, timesteps=0, output_root=tmp_path)


@pytest.mark.parametrize(
  ("callback_name", "callback", "message"),
  (
    (
      "guarded_selection_annotation",
      guarded_selection_annotation,
      "dynamic conditional is outside",
    ),
    (
      "selection_else_annotation",
      selection_else_annotation,
      "aggregate selection cannot have an else",
    ),
    (
      "repeated_clause_loop_annotation",
      repeated_clause_loop_annotation,
      "multiple dynamic loops over one callback clause",
    ),
    (
      "unrelated_fallback_annotation",
      unrelated_fallback_annotation,
      "dynamic conditional is outside",
    ),
    (
      "boolean_crossed_fallback_annotation",
      boolean_crossed_fallback_annotation,
      "dynamic conditional is outside",
    ),
    (
      "negated_crossed_fallback_annotation",
      negated_crossed_fallback_annotation,
      "dynamic conditional is outside",
    ),
    (
      "nonterminal_selection_annotation",
      nonterminal_selection_annotation,
      "aggregate selection must be the final statement",
    ),
    (
      "loop_else_annotation",
      loop_else_annotation,
      "for-else control flow is unsupported",
    ),
    (
      "nested_nonterminal_selection_annotation",
      nested_nonterminal_selection_annotation,
      "must occur directly in its dynamic loop body",
    ),
    (
      "loop_carried_candidate_annotation",
      loop_carried_candidate_annotation,
      "carries unrecognized state between dynamic loop iterations",
    ),
  ),
)
def test_callback_rejects_unproven_control_flow(
  tmp_path: Path,
  callback_name: str,
  callback: object,
  message: str,
) -> None:
  source = SourceProgram(
    rules=(
      SourceRule(
        text=f'output(X):{callback_name} <- input(X)',
        name='unproven-callback',
        head_predicate='output',
        head_terms=('X',),
        head_annotation=callback_name,
        head_lower=0.0,
        head_upper=1.0,
        delay=0,
        clauses=(SourceClause('input', ('X',), 0.0, 1.0),),
      ),
    ),
    facts=(SourceFact('input(a)', 'input-a', 'input', ('a',), 0.8, 1.0, 0, 0, True),),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=((callback_name, callback),),
  )

  with pytest.raises(UnsupportedAnnotationSemantics, match=message):
    compile_source(source, timesteps=0, output_root=tmp_path)


@pytest.mark.parametrize(
  ("callback_name", "callback", "message"),
  (
    (
      "boolean_lookup_annotation",
      boolean_lookup_annotation,
      "dynamic conditional is outside",
    ),
    (
      "nonterminal_lookup_annotation",
      nonterminal_lookup_annotation,
      "aligned lookup must be the final statement",
    ),
    (
      "stale_lookup_sentinel_annotation",
      stale_lookup_sentinel_annotation,
      "lookup sentinels must be reset in the immediately enclosing driver loop",
    ),
    (
      "self_equal_lookup_annotation",
      self_equal_lookup_annotation,
      "lookup equality must compare the innermost row with one enclosing row",
    ),
  ),
)
def test_callback_rejects_unproven_lookup_control_flow(
  tmp_path: Path,
  callback_name: str,
  callback: object,
  message: str,
) -> None:
  source = SourceProgram(
    rules=(
      SourceRule(
        text=f'output(X):{callback_name} <- input(X), context(X)',
        name='unproven-lookup',
        head_predicate='output',
        head_terms=('X',),
        head_annotation=callback_name,
        head_lower=0.0,
        head_upper=1.0,
        delay=0,
        clauses=(
          SourceClause('input', ('X',), 0.0, 1.0),
          SourceClause('context', ('X',), 0.0, 1.0),
        ),
      ),
    ),
    facts=(
      SourceFact('input(a)', 'input-a', 'input', ('a',), 0.8, 1.0, 0, 0, True),
      SourceFact('context(a)', 'context-a', 'context', ('a',), 0.7, 0.9, 0, 0, True),
    ),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=((callback_name, callback),),
  )

  with pytest.raises(UnsupportedAnnotationSemantics, match=message):
    compile_source(source, timesteps=0, output_root=tmp_path)


def test_callback_rejects_row_value_escaping_a_sequential_loop(
  tmp_path: Path,
) -> None:
  source = SourceProgram(
    rules=(
      SourceRule(
        text='output(H):sequential_loop_escape_annotation <- p(Y), q(Y), anchor(H)',
        name='sequential-loop-escape',
        head_predicate='output',
        head_terms=('H',),
        head_annotation='sequential_loop_escape_annotation',
        head_lower=0.0,
        head_upper=1.0,
        delay=0,
        clauses=(
          SourceClause('p', ('Y',), 0.0, 1.0),
          SourceClause('q', ('Y',), 0.0, 1.0),
          SourceClause('anchor', ('H',), 0.0, 1.0),
        ),
      ),
    ),
    facts=(
      SourceFact('p(a)', 'p-a', 'p', ('a',), 0.9, 1.0, 0, 0, True),
      SourceFact('p(b)', 'p-b', 'p', ('b',), 0.1, 1.0, 0, 0, True),
      SourceFact('q(a)', 'q-a', 'q', ('a',), 0.8, 1.0, 0, 0, True),
      SourceFact('q(b)', 'q-b', 'q', ('b',), 0.7, 1.0, 0, 0, True),
      SourceFact('anchor(h)', 'anchor', 'anchor', ('h',), 1.0, 1.0, 0, 0, True),
    ),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(
      ('sequential_loop_escape_annotation', sequential_loop_escape_annotation),
    ),
  )

  with pytest.raises(
    UnsupportedAnnotationSemantics,
    match="row-derived local 'saved_lower' escapes its dynamic loop",
  ):
    compile_source(source, timesteps=0, output_root=tmp_path)


def test_seed_rank_tracks_predicate_map_admission_not_source_index(
  tmp_path: Path,
) -> None:
  rule = SourceRule(
    text='output(H):six_argument_annotation <- input(Y), anchor(H)',
    name='admission-order',
    head_predicate='output',
    head_terms=('H',),
    head_annotation='six_argument_annotation',
    head_lower=0.0,
    head_upper=1.0,
    delay=0,
    clauses=(
      SourceClause('input', ('Y',), 0.1, 1.0),
      SourceClause('anchor', ('H',), 1.0, 1.0),
    ),
  )
  source = SourceProgram(
    rules=(rule,),
    facts=(
      SourceFact('input(a)', 'a-late', 'input', ('a',), 0.5, 0.8, 5, 5, False),
      SourceFact('input(b)', 'b-early', 'input', ('b',), 0.5, 0.9, 0, 5, False),
      SourceFact('anchor(h)', 'anchor', 'anchor', ('h',), 1, 1, 0, 0, True),
    ),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(('six_argument_annotation', six_argument_annotation),),
  )

  plan = compile_source(source, timesteps=5, output_root=tmp_path)
  input_output = next(output for output in plan.outputs if output.predicate == 'input')
  input_index = int(input_output.relation.removeprefix('PyReasonRelation'))
  rows = [
    line.split(',') for line in (plan.data_dir / f'seed_{input_index}.csv').read_text().splitlines()
  ]
  admission_rows = [
    line.split(',')
    for line in (plan.data_dir / f'admission_rank_{input_index}.csv').read_text().splitlines()
  ]
  by_symbol_time = {(int(row[0]), int(row[1])): int(row[2]) for row in rows}
  rank_by_symbol = {int(row[0]): int(row[1]) for row in admission_rows}
  symbols = {symbol: identifier for identifier, symbol in plan.symbol_names.items()}
  assert by_symbol_time[(symbols['b'], 0)] == 1
  assert by_symbol_time[(symbols['a'], 5)] == 2
  assert rank_by_symbol == {symbols['b']: 1, symbols['a']: 2}


def test_effective_source_preserves_pyreason_clause_reorder_mapping(
  tmp_path: Path,
) -> None:
  source = SourceProgram(
    rules=(
      SourceRule(
        text='out(X) <- edge(X,Y), node(X)',
        name='mixed',
        head_predicate='out',
        head_terms=('X',),
        head_annotation=None,
        head_lower=1,
        head_upper=1,
        delay=0,
        clauses=(
          SourceClause('edge', ('X', 'Y'), 1, 1),
          SourceClause('node', ('X',), 1, 1),
        ),
        weights=(0.25, 0.75),
      ),
    ),
    facts=(
      SourceFact('edge(a,b)', 'edge', 'edge', ('a', 'b'), 1, 1, 0, 0, True),
      SourceFact('node(a)', 'node', 'node', ('a',), 1, 1, 0, 0, True),
    ),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(),
    node_domain=('a', 'b'),
    edge_domain=(('a', 'b'),),
    reorder_clauses_node_first=True,
  )

  plan = compile_source(source, timesteps=0, output_root=tmp_path)
  assert plan.effective_source is not None
  effective = plan.effective_source.rules[0]
  assert [clause.predicate for clause in effective.clauses] == ['node', 'edge']
  assert effective.clause_source_positions == (2, 1)
  # PyReason reorders clauses/thresholds, but leaves the weight array untouched.
  assert effective.weights == (0.25, 0.75)


def test_duplicate_display_names_do_not_alias_callback_compilation(tmp_path: Path) -> None:
  clause = SourceClause('input', ('x',), 0.0, 1.0)
  callback_rule = SourceRule(
    text='callback(x):six_argument_annotation <- input(x)',
    name='duplicate',
    head_predicate='callback',
    head_terms=('x',),
    head_annotation='six_argument_annotation',
    head_lower=0.0,
    head_upper=1.0,
    delay=0,
    clauses=(clause,),
  )
  constant_rule = SourceRule(
    text='constant(x):[0.2,0.3] <- input(x)',
    name='duplicate',
    head_predicate='constant',
    head_terms=('x',),
    head_annotation=None,
    head_lower=0.2,
    head_upper=0.3,
    delay=0,
    clauses=(clause,),
  )
  source = SourceProgram(
    rules=(callback_rule, constant_rule),
    facts=(SourceFact('input(a)', 'input-a', 'input', ('a',), 1, 1, 0, 0, True),),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(('six_argument_annotation', six_argument_annotation),),
  )

  plan = compile_source(source, timesteps=0, output_root=tmp_path)
  callback = next(rule for rule in plan.program.rules if rule.name == 'PyReasonRule0_duplicate')
  callback_candidate = next(
    rule
    for rule in plan.program.rules
    if rule.name == 'PyReasonRule0_duplicate__aggregate_candidate'
  )
  constant = next(rule for rule in plan.program.rules if rule.name == 'PyReasonRule1_duplicate')

  assert not any(isinstance(clause, Let) for clause in callback.body)
  assert any(isinstance(clause, Let) for clause in callback_candidate.body)
  assert not any(isinstance(clause, Let) for clause in constant.body)


def test_callback_provenance_rejects_missing_group_event_semantics(
  tmp_path: Path,
) -> None:
  clause = SourceClause('input', ('x',), 0.0, 1.0)
  source = SourceProgram(
    rules=(
      SourceRule(
        text='output(x):six_argument_annotation <- input(x)',
        name='callback',
        head_predicate='output',
        head_terms=('x',),
        head_annotation='six_argument_annotation',
        head_lower=0.0,
        head_upper=1.0,
        delay=0,
        clauses=(clause,),
      ),
    ),
    facts=(SourceFact('input(a)', 'input-a', 'input', ('a',), 1, 1, 0, 0, True),),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(('six_argument_annotation', six_argument_annotation),),
  )

  with pytest.raises(
    UnsupportedAnnotationSemantics,
    match='logical aggregate-change event',
  ):
    compile_source(
      source,
      timesteps=0,
      output_root=tmp_path,
      provenance_demands=(DemandSeed('output', ('a',), 0),),
    )


def test_ground_rule_mode_is_inert_without_graph_matching_terms(tmp_path: Path) -> None:
  source = SourceProgram(
    rules=(),
    facts=(SourceFact('p(a)', 'p-a', 'p', ('a',), 1, 1, 0, 0, False),),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(),
    settings=(('allow_ground_rules', True),),
  )

  assert compile_source(source, timesteps=0, output_root=tmp_path) is not None


def test_ground_rule_mode_rejects_a_term_matching_a_graph_node(tmp_path: Path) -> None:
  source = SourceProgram(
    rules=(
      SourceRule(
        text='q(a) <- p(a)',
        name='ground-a',
        head_predicate='q',
        head_terms=('a',),
        head_annotation=None,
        head_lower=1,
        head_upper=1,
        delay=0,
        clauses=(SourceClause('p', ('a',), 1, 1),),
      ),
    ),
    facts=(SourceFact('p(a)', 'p-a', 'p', ('a',), 1, 1, 0, 0, False),),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(),
    settings=(('allow_ground_rules', True),),
    node_domain=('a',),
  )

  with pytest.raises(UnsupportedAnnotationSemantics, match='allow_ground_rules=True'):
    compile_source(source, timesteps=0, output_root=tmp_path)
