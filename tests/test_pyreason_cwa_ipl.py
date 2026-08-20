from dataclasses import replace
from pathlib import Path

import pytest

from srdatalog import (
  ScalarAnd,
  ScalarCompare,
  ScalarOr,
  ScalarVar,
  build_project,
  compile_to_hir,
  float32_to_u32,
)
from srdatalog.dsl import Atom, Filter, Rule
from srdatalog.pyreason import (
  NativePlan,
  SourceClause,
  SourceFact,
  SourceProgram,
  SourceRule,
  UnsupportedAnnotationSemantics,
  compile_source,
)


def _fact(predicate: str, arguments: tuple[str, ...], lower: float, upper: float) -> SourceFact:
  return SourceFact(
    text="",
    name=f"{predicate}-fact",
    predicate=predicate,
    arguments=arguments,
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
  terms: tuple[str, ...] = ("X",),
  body_lower: float = 0.0,
  body_upper: float = 0.0,
) -> SourceRule:
  return SourceRule(
    text="",
    name=name,
    head_predicate=head,
    head_terms=terms,
    head_annotation=None,
    head_lower=1.0,
    head_upper=1.0,
    delay=0,
    clauses=(SourceClause(body, terms, body_lower, body_upper),),
  )


def _max_callback(
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


def _named_rule(plan: NativePlan, name: str) -> Rule:
  for rule in plan.program.rules:
    if isinstance(rule, Rule) and rule.name == name:
      return rule
  raise AssertionError(f"missing rule {name!r}")


def _first_atom(rule: Rule) -> Atom:
  clause = rule.body[0]
  assert isinstance(clause, Atom)
  return clause


def _first_filter(rule: Rule) -> Filter:
  result = next(clause for clause in rule.body if isinstance(clause, Filter))
  assert isinstance(result, Filter)
  return result


def test_closed_world_read_view_materializes_bottom_and_guard_interprets_false(
  tmp_path: Path,
) -> None:
  source = SourceProgram(
    rules=(
      _rule("closed-read", "closed_hit", "closed"),
      _rule("open-read", "open_hit", "open"),
    ),
    facts=(
      _fact("closed", ("a",), 0.0, 1.0),
      _fact("open", ("a",), 0.0, 1.0),
    ),
    graphml_path=None,
    closed_world_predicates=frozenset({"closed"}),
    annotation_functions=(),
    node_domain=("a", "b"),
  )

  plan = compile_source(source, timesteps=0, output_root=tmp_path)
  assert plan is not None
  range_rule = next(rule for rule in plan.program.rules if rule.name.startswith("PyReasonCwaRange"))

  # The total active range is a compiler-generated lattice read view. Actual
  # facts/rules remain in the public logical relation and flow into this view.
  assert [type(clause) for clause in range_rule.body] == [Atom]
  assert _first_atom(range_rule).rel == "PyReasonNodeTimeDomain"
  assert range_rule.head.args[-2].const_value == float32_to_u32(0.0)
  assert range_rule.head.args[-1].const_value == float32_to_u32(1.0)

  closed_read = _named_rule(plan, "PyReasonRule0_closed_read")
  open_read = _named_rule(plan, "PyReasonRule1_open_read")
  assert _first_atom(closed_read).rel == range_rule.head.rel
  assert _first_atom(open_read).rel.startswith("PyReasonRelation")
  assert isinstance(_first_filter(closed_read).expression, ScalarOr)
  assert isinstance(_first_filter(open_read).expression, ScalarAnd)
  eligible_atoms = [
    clause
    for clause in closed_read.body
    if isinstance(clause, Atom) and clause.rel.startswith("PyReasonCwaEligible")
  ]
  assert len(eligible_atoms) == 1
  assert (plan.data_dir / "cwa_time_leq.csv").read_text().splitlines() == ["0,0"]
  closed_choices = _first_filter(closed_read).expression
  assert isinstance(closed_choices, ScalarOr)
  assert [type(choice) for choice in closed_choices.arguments] == [
    ScalarAnd,
    ScalarAnd,
    ScalarCompare,
  ]
  crossed = closed_choices.arguments[-1]
  assert isinstance(crossed, ScalarCompare)
  assert crossed.operator == ">"
  assert (plan.data_dir / "cwa_node_time_domain.csv").read_text().splitlines() == [
    "0,0",
    "1,0",
  ]
  generated = build_project(
    plan.program,
    project_name="ClosedWorldShape",
    cache_base=str(tmp_path / "generated"),
  )
  assert Path(generated["main"]).is_file()


def test_bound_closed_world_lookup_does_not_restrict_to_labeled_keys(
  tmp_path: Path,
) -> None:
  rule = SourceRule(
    text="bad(X) <- domain(X), ~closed(X)",
    name="bound-closed-read",
    head_predicate="bad",
    head_terms=("X",),
    head_annotation=None,
    head_lower=1.0,
    head_upper=1.0,
    delay=0,
    clauses=(
      SourceClause("domain", ("X",), 1.0, 1.0),
      SourceClause("closed", ("X",), 0.0, 0.0),
    ),
  )
  source = SourceProgram(
    rules=(rule,),
    facts=(
      _fact("domain", ("b",), 1.0, 1.0),
      _fact("closed", ("a",), 1.0, 1.0),
    ),
    graphml_path=None,
    closed_world_predicates=frozenset({"closed"}),
    annotation_functions=(),
    node_domain=("a", "b"),
  )

  plan = compile_source(source, timesteps=0, output_root=tmp_path)

  assert plan is not None
  compiled = _named_rule(plan, "PyReasonRule0_bound_closed_read")
  assert all(
    not (isinstance(clause, Atom) and clause.rel.startswith("PyReasonCwaEligible"))
    for clause in compiled.body
  )


def test_cwa_seed_world_is_initialized_before_default_rule_rounds(
  tmp_path: Path,
) -> None:
  source = SourceProgram(
    rules=(
      _rule("make-p-idb", "p", "seed", body_lower=1.0, body_upper=1.0),
      SourceRule(
        text="bad(X) <- r(X), p(X):[0,0]",
        name="default-read",
        head_predicate="bad",
        head_terms=("X",),
        head_annotation=None,
        head_lower=1.0,
        head_upper=1.0,
        delay=0,
        clauses=(
          SourceClause("r", ("X",), 1.0, 1.0),
          SourceClause("p", ("X",), 0.0, 0.0),
        ),
      ),
    ),
    facts=(
      replace(_fact("p", ("a",), 1.0, 1.0), static=False),
      replace(_fact("r", ("a",), 1.0, 1.0), static=False),
    ),
    graphml_path=None,
    closed_world_predicates=frozenset({"p"}),
    annotation_functions=(),
    node_domain=("a",),
  )

  plan = compile_source(source, timesteps=0, output_root=tmp_path)
  strata = compile_to_hir(plan.program).strata
  seed_stratum = next(
    index
    for index, stratum in enumerate(strata)
    if any(rule.name.startswith("PyReasonCwaSeed") for rule in stratum.stratum_rules)
  )
  default_stratum = next(
    index
    for index, stratum in enumerate(strata)
    if any(rule.name == "PyReasonRule1_default_read__group" for rule in stratum.stratum_rules)
  )
  assert seed_stratum < default_stratum


def test_partially_bound_binary_cwa_lookup_uses_graph_neighbors(
  tmp_path: Path,
) -> None:
  rule = SourceRule(
    text="out(Y) <- anchor(X), ~closed(X,Y)",
    name="partially-bound-edge-read",
    head_predicate="out",
    head_terms=("Y",),
    head_annotation=None,
    head_lower=1.0,
    head_upper=1.0,
    delay=0,
    clauses=(
      SourceClause("anchor", ("X",), 1.0, 1.0),
      SourceClause("closed", ("X", "Y"), 0.0, 0.0),
    ),
  )
  source = SourceProgram(
    rules=(rule,),
    facts=(
      _fact("anchor", ("a",), 1.0, 1.0),
      _fact("closed", ("a", "b"), 1.0, 1.0),
    ),
    graphml_path=None,
    closed_world_predicates=frozenset({"closed"}),
    annotation_functions=(),
    node_domain=("a", "b", "c"),
    edge_domain=(("a", "b"), ("a", "c")),
  )

  plan = compile_source(source, timesteps=0, output_root=tmp_path)

  compiled = _named_rule(plan, "PyReasonRule0_partially_bound_edge_read")
  assert all(
    not (isinstance(clause, Atom) and clause.rel.startswith("PyReasonCwaEligible"))
    for clause in compiled.body
  )


def test_every_source_rule_uses_the_same_staged_commit_boundary(
  tmp_path: Path,
) -> None:
  source = SourceProgram(
    rules=(
      _rule("make-p", "p", "base", body_lower=1.0, body_upper=1.0),
      _rule("read-default", "bad", "p"),
    ),
    facts=(_fact("base", ("a",), 1.0, 1.0),),
    graphml_path=None,
    closed_world_predicates=frozenset({"p"}),
    annotation_functions=(),
    node_domain=("a",),
  )

  plan = compile_source(source, timesteps=0, output_root=tmp_path)

  assert plan is not None
  output_relations = {output.predicate: output.relation for output in plan.outputs}
  hir = compile_to_hir(plan.program)
  synchronized = next(
    stratum
    for stratum in hir.strata
    if stratum.is_recursive and "PyReasonRoundSync" in stratum.scc_members
  )
  assert {output_relations["p"], output_relations["bad"]} <= synchronized.scc_members
  # Constant heads and callback heads must both take producer -> commit.  If a
  # constant rule writes its logical head directly, it becomes visible one
  # target iteration before a grouped callback and violates PyReason's queued
  # update barrier.
  names = {rule.name for stratum in hir.strata for rule in stratum.stratum_rules}
  assert {
    "PyReasonRule0_make_p__group",
    "PyReasonRule0_make_p__project",
    "PyReasonRule1_read_default__group",
    "PyReasonRule1_read_default__project",
  } <= names


def test_idb_closed_world_read_uses_compiler_read_view(tmp_path: Path) -> None:
  source = SourceProgram(
    rules=(
      _rule("derive-middle", "middle", "base", body_lower=1.0, body_upper=1.0),
      _rule("read-middle", "result", "middle"),
    ),
    facts=(_fact("base", ("a",), 1.0, 1.0),),
    graphml_path=None,
    closed_world_predicates=frozenset({"middle"}),
    annotation_functions=(),
    node_domain=("a",),
  )

  plan = compile_source(source, timesteps=0, output_root=tmp_path)

  assert plan is not None
  range_rule = next(rule for rule in plan.program.rules if rule.name.startswith("PyReasonCwaRange"))
  assert _first_atom(_named_rule(plan, "PyReasonRule1_read_middle")).rel == range_rule.head.rel


def test_binary_closed_world_range_uses_edge_time_active_domain(tmp_path: Path) -> None:
  source = SourceProgram(
    rules=(
      _rule(
        "missing-edge",
        "missing",
        "edge_value",
        terms=("Source", "Target"),
      ),
    ),
    facts=(),
    graphml_path=None,
    closed_world_predicates=frozenset({"edge_value"}),
    annotation_functions=(),
    node_domain=("a", "b"),
    edge_domain=(("a", "b"),),
  )

  plan = compile_source(source, timesteps=1, output_root=tmp_path)

  assert plan is not None
  range_rule = next(rule for rule in plan.program.rules if rule.name.startswith("PyReasonCwaRange"))
  assert _first_atom(range_rule).rel == "PyReasonEdgeTimeDomain"
  assert (plan.data_dir / "cwa_edge_time_domain.csv").read_text().splitlines() == [
    "0,1,0",
    "0,1,1",
  ]


def test_closed_world_cycle_uses_logical_state_and_recursive_read_view(
  tmp_path: Path,
) -> None:
  source = SourceProgram(
    rules=(
      _rule("read-right", "left", "right"),
      _rule("derive-right", "right", "left", body_lower=1.0, body_upper=1.0),
    ),
    facts=(),
    graphml_path=None,
    closed_world_predicates=frozenset({"right"}),
    annotation_functions=(),
    node_domain=("a",),
  )

  plan = compile_source(source, timesteps=0, output_root=tmp_path)

  assert plan is not None
  range_rule = next(rule for rule in plan.program.rules if rule.name.startswith("PyReasonCwaRange"))
  closed_read = _named_rule(plan, "PyReasonRule0_read_right")
  recursive_write = _named_rule(plan, "PyReasonRule1_derive_right")
  assert _first_atom(closed_read).rel == range_rule.head.rel
  assert recursive_write.head.rel != range_rule.head.rel
  actual_bridge = next(
    rule for rule in plan.program.rules if rule.name.startswith("PyReasonCwaActual")
  )
  assert actual_bridge.head.rel == range_rule.head.rel
  assert _first_atom(actual_bridge).rel == recursive_write.head.rel
  assert isinstance(_first_filter(closed_read).expression, ScalarOr)


def test_positive_closed_world_recursion_remains_an_ordinary_read(
  tmp_path: Path,
) -> None:
  source = SourceProgram(
    rules=(
      _rule(
        "positive-recursion",
        "analystAt",
        "analystAt",
        body_lower=0.25,
        body_upper=1.0,
      ),
    ),
    facts=(replace(_fact("analystAt", ("a",), 1.0, 1.0), static=False),),
    graphml_path=None,
    closed_world_predicates=frozenset({"analystAt"}),
    annotation_functions=(),
    node_domain=("a",),
  )

  plan = compile_source(source, timesteps=0, output_root=tmp_path)

  assert plan is not None
  positive_read = _first_atom(_named_rule(plan, "PyReasonRule0_positive_recursion"))
  assert positive_read.rel.startswith("PyReasonRelation")
  assert all(not rule.name.startswith("PyReasonCwa") for rule in plan.program.rules)


def test_callback_rejects_disappearing_closed_world_default_member(
  tmp_path: Path,
) -> None:
  source = SourceProgram(
    rules=(
      SourceRule(
        text="out(X):_max_callback <- closed(X):[0,0]",
        name="callback-default-read",
        head_predicate="out",
        head_terms=("X",),
        head_annotation="_max_callback",
        head_lower=0.0,
        head_upper=1.0,
        delay=0,
        clauses=(SourceClause("closed", ("X",), 0.0, 0.0),),
      ),
      _rule(
        "derive-closed",
        "closed",
        "trigger",
        body_lower=1.0,
        body_upper=1.0,
      ),
    ),
    facts=(
      SourceFact("closed(a):[0,1]", "closed-a", "closed", ("a",), 0, 1, 0, 0, False),
      _fact("trigger", ("a",), 1.0, 1.0),
    ),
    graphml_path=None,
    closed_world_predicates=frozenset({"closed"}),
    annotation_functions=(("_max_callback", _max_callback),),
    node_domain=("a",),
  )

  with pytest.raises(
    UnsupportedAnnotationSemantics,
    match="cannot retain a closed-world default member",
  ):
    compile_source(source, timesteps=0, output_root=tmp_path)


def test_positive_closed_world_read_does_not_wait_for_default_eligibility(
  tmp_path: Path,
) -> None:
  source = SourceProgram(
    rules=(
      _rule(
        "positive-read",
        "copied",
        "closed",
        body_lower=0.25,
        body_upper=1.0,
      ),
      _rule("default-read", "missing", "closed"),
    ),
    facts=(_fact("closed", ("a",), 1.0, 1.0),),
    graphml_path=None,
    closed_world_predicates=frozenset({"closed"}),
    annotation_functions=(),
    node_domain=("a",),
  )

  plan = compile_source(source, timesteps=0, output_root=tmp_path)

  assert plan is not None
  positive = _named_rule(plan, "PyReasonRule0_positive_read")
  default = _named_rule(plan, "PyReasonRule1_default_read")
  assert all(
    not (isinstance(clause, Atom) and clause.rel.startswith("PyReasonCwaEligible"))
    for clause in positive.body
  )
  assert any(
    isinstance(clause, Atom) and clause.rel.startswith("PyReasonCwaEligible")
    for clause in default.body
  )


def test_late_fact_switches_unbound_cwa_grounding_domain_by_time(
  tmp_path: Path,
) -> None:
  source = SourceProgram(
    rules=(_rule("default-read", "missing", "p"),),
    facts=(SourceFact("", "p-late", "p", ("a",), 1.0, 1.0, 1, 1, False),),
    graphml_path=None,
    closed_world_predicates=frozenset({"p"}),
    annotation_functions=(),
    node_domain=("a", "b"),
  )

  plan = compile_source(source, timesteps=1, output_root=tmp_path)

  assert (plan.data_dir / "cwa_eligible_1.csv").read_text().splitlines() == [
    "0,0",
    "1,0",
    "0,1",
  ]


def test_future_fact_component_is_absent_from_earlier_cwa_domain(
  tmp_path: Path,
) -> None:
  source = SourceProgram(
    rules=(_rule("default-read", "missing", "p"),),
    facts=(SourceFact("", "future-q", "q", ("b",), 1.0, 1.0, 1, 1, False),),
    graphml_path=None,
    closed_world_predicates=frozenset({"p"}),
    annotation_functions=(),
    node_domain=("a", "b"),
    node_birth_times=(("a", 0), ("b", 1)),
  )

  plan = compile_source(source, timesteps=1, output_root=tmp_path)

  assert (plan.data_dir / "cwa_node_time_domain.csv").read_text().splitlines() == [
    "0,0",
    "0,1",
    "1,1",
  ]


def test_idb_first_label_with_unbound_cwa_read_is_rejected(
  tmp_path: Path,
) -> None:
  source = SourceProgram(
    rules=(
      _rule("derive-p", "p", "base", body_lower=1.0, body_upper=1.0),
      SourceRule(
        text="",
        name="wait-then-default",
        head_predicate="bad",
        head_terms=("Y",),
        head_annotation=None,
        head_lower=1.0,
        head_upper=1.0,
        delay=0,
        clauses=(
          SourceClause("p", ("X",), 1.0, 1.0),
          SourceClause("p", ("Y",), 0.0, 0.0),
        ),
      ),
    ),
    facts=(_fact("base", ("a",), 1.0, 1.0),),
    graphml_path=None,
    closed_world_predicates=frozenset({"p"}),
    annotation_functions=(),
    node_domain=("a", "b"),
  )

  with pytest.raises(
    UnsupportedAnnotationSemantics,
    match="source-round grounding-domain relation",
  ):
    compile_source(source, timesteps=0, output_root=tmp_path)


def test_label_only_idb_update_is_real_predicate_map_presence(tmp_path: Path) -> None:
  label_only = replace(
    _rule("mention-p", "p", "base", body_lower=1.0, body_upper=1.0),
    head_lower=0.0,
    head_upper=1.0,
  )
  source = SourceProgram(
    rules=(
      label_only,
      _rule("default-read", "missing", "p"),
    ),
    facts=(
      replace(_fact("p", ("a",), 1.0, 1.0), static=False),
      _fact("base", ("b",), 1.0, 1.0),
    ),
    graphml_path=None,
    closed_world_predicates=frozenset({"p"}),
    annotation_functions=(),
    node_domain=("a", "b"),
  )

  plan = compile_source(source, timesteps=0, output_root=tmp_path)

  output = next(item for item in plan.outputs if item.predicate == "p")
  assert output.suppress_bottom is False
  eligibility = next(
    rule
    for rule in plan.program.rules
    if rule.name.startswith("PyReasonCwaEligible") and rule.name.endswith("Derived")
  )
  assert not any(isinstance(clause, Filter) for clause in eligibility.body)


def test_phase_sensitive_cwa_temporal_initialization_is_rejected(
  tmp_path: Path,
) -> None:
  delayed = SourceProgram(
    rules=(
      SourceRule(
        text="",
        name="delayed-p",
        head_predicate="p",
        head_terms=("X",),
        head_annotation=None,
        head_lower=1.0,
        head_upper=1.0,
        delay=1,
        clauses=(SourceClause("base", ("X",), 1.0, 1.0),),
      ),
      _rule("default-read", "bad", "p"),
    ),
    facts=(
      _fact("base", ("a",), 1.0, 1.0),
      replace(_fact("p", ("a",), 1.0, 1.0), static=False),
    ),
    graphml_path=None,
    closed_world_predicates=frozenset({"p"}),
    annotation_functions=(),
    node_domain=("a", "b"),
  )
  persistent = replace(
    delayed,
    rules=(_rule("default-read", "bad", "p"),),
    settings=(("persistent", True),),
  )

  with pytest.raises(UnsupportedAnnotationSemantics, match="time-stratified worlds"):
    compile_source(delayed, timesteps=1, output_root=tmp_path)
  with pytest.raises(UnsupportedAnnotationSemantics, match="phase-stratified"):
    compile_source(persistent, timesteps=1, output_root=tmp_path)


def test_bound_cwa_read_still_rejects_delayed_phase_skew(tmp_path: Path) -> None:
  bound_default = SourceRule(
    text="bad(X) <- domain(X), ~p(X)",
    name="bound-default-read",
    head_predicate="bad",
    head_terms=("X",),
    head_annotation=None,
    head_lower=1.0,
    head_upper=1.0,
    delay=0,
    clauses=(
      SourceClause("domain", ("X",), 1.0, 1.0),
      SourceClause("p", ("X",), 0.0, 0.0),
    ),
  )
  delayed_p = SourceRule(
    text="p(X) <-1 seed(X)",
    name="delayed-p",
    head_predicate="p",
    head_terms=("X",),
    head_annotation=None,
    head_lower=1.0,
    head_upper=1.0,
    delay=1,
    clauses=(SourceClause("seed", ("X",), 1.0, 1.0),),
  )
  source = SourceProgram(
    rules=(delayed_p, bound_default),
    facts=(
      _fact("seed", ("a",), 1.0, 1.0),
      _fact("domain", ("a",), 1.0, 1.0),
    ),
    graphml_path=None,
    closed_world_predicates=frozenset({"p"}),
    annotation_functions=(),
    node_domain=("a",),
  )

  with pytest.raises(UnsupportedAnnotationSemantics, match="time-stratified worlds"):
    compile_source(source, timesteps=1, output_root=tmp_path)


def test_static_freeze_collisions_are_rejected(tmp_path: Path) -> None:
  static_p = _fact("p", ("a",), 0.8, 1.0)
  rule_collision = SourceProgram(
    rules=(_rule("update-p", "p", "q", body_lower=1.0, body_upper=1.0),),
    facts=(static_p, _fact("q", ("a",), 1.0, 1.0)),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(),
    node_domain=("a",),
  )
  fact_collision = replace(
    rule_collision,
    rules=(),
    facts=(
      static_p,
      SourceFact("", "p-later", "p", ("a",), 0.0, 0.2, 0, 0, False),
    ),
  )

  with pytest.raises(UnsupportedAnnotationSemantics, match="static-key write guard"):
    compile_source(rule_collision, timesteps=0, output_root=tmp_path)
  with pytest.raises(UnsupportedAnnotationSemantics, match="source-order-sensitive"):
    compile_source(fact_collision, timesteps=0, output_root=tmp_path)


def test_identical_static_fact_declarations_are_idempotent(tmp_path: Path) -> None:
  first = _fact("p", ("a",), 0.8, 1.0)
  duplicate = replace(first, text="p(a) duplicate", name="duplicate")
  source = SourceProgram(
    rules=(),
    facts=(first, duplicate),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(),
    node_domain=("a",),
  )

  assert compile_source(source, timesteps=0, output_root=tmp_path) is not None


def test_binary_head_requires_existing_edge_and_invention_is_rejected(
  tmp_path: Path,
) -> None:
  binary = SourceRule(
    text="p(X,Y) <- left(X), right(Y)",
    name="existing-edge-only",
    head_predicate="p",
    head_terms=("X", "Y"),
    head_annotation=None,
    head_lower=1.0,
    head_upper=1.0,
    delay=0,
    clauses=(
      SourceClause("left", ("X",), 1.0, 1.0),
      SourceClause("right", ("Y",), 1.0, 1.0),
    ),
  )
  source = SourceProgram(
    rules=(binary,),
    facts=(
      _fact("left", ("a",), 1.0, 1.0),
      _fact("right", ("b",), 1.0, 1.0),
    ),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(),
    node_domain=("a", "b"),
    edge_domain=(("a", "b"),),
  )

  plan = compile_source(source, timesteps=0, output_root=tmp_path)
  compiled = _named_rule(plan, "PyReasonRule0_existing_edge_only")
  assert any(
    isinstance(clause, Atom) and clause.rel == "PyReasonActiveEdgeTime" for clause in compiled.body
  )
  assert (plan.data_dir / "active_edge_time.csv").read_text().splitlines() == [
    "0,1,0",
  ]

  with pytest.raises(UnsupportedAnnotationSemantics, match="binary infer_edges"):
    compile_source(
      replace(source, rules=(replace(binary, infer_edges=True),)),
      timesteps=0,
      output_root=tmp_path,
    )


def test_inconsistency_pair_is_rejected_until_updates_co_commit(
  tmp_path: Path,
) -> None:
  source = SourceProgram(
    rules=(),
    facts=(_fact("present", ("a",), 0.25, 0.5),),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(),
    inconsistent_predicates=(("present", "absent"),),
  )

  with pytest.raises(
    UnsupportedAnnotationSemantics,
    match="atomic, one-hop co-commit",
  ):
    compile_source(source, timesteps=0, output_root=tmp_path)


def test_abort_on_inconsistency_is_rejected_explicitly(tmp_path: Path) -> None:
  source = SourceProgram(
    rules=(),
    facts=(_fact("node_value", ("a",), 1.0, 1.0),),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(),
    settings=(("abort_on_inconsistency", True),),
  )

  with pytest.raises(UnsupportedAnnotationSemantics, match="update-time short-circuit"):
    compile_source(source, timesteps=0, output_root=tmp_path)


def test_ordinary_body_guard_rejects_raw_crossed_intervals(tmp_path: Path) -> None:
  source = SourceProgram(
    rules=(
      _rule(
        "copy-consistent",
        "output",
        "input",
        body_lower=0.25,
        body_upper=1.0,
      ),
    ),
    facts=(_fact("input", ("a",), 0.5, 0.75),),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(),
  )

  plan = compile_source(source, timesteps=0, output_root=tmp_path)
  assert plan is not None
  guard = next(
    clause
    for clause in _named_rule(plan, "PyReasonRule0_copy_consistent").body
    if isinstance(clause, Filter)
  )
  assert isinstance(guard.expression, ScalarAnd)
  comparisons = guard.expression.arguments
  assert isinstance(comparisons[0], ScalarCompare)
  assert comparisons[0].operator == "<="
  assert isinstance(comparisons[0].left, ScalarVar)
  assert isinstance(comparisons[0].right, ScalarVar)
  assert comparisons[0].left.name.endswith("_lower_0")
  assert comparisons[0].right.name.endswith("_upper_0")


def test_broad_ordinary_body_guard_accepts_repaired_conflict(tmp_path: Path) -> None:
  source = SourceProgram(
    rules=(_rule("copy-unknown", "output", "input", body_upper=1.0),),
    facts=(_fact("input", ("a",), 0.5, 0.75),),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(),
  )

  plan = compile_source(source, timesteps=0, output_root=tmp_path)
  assert plan is not None
  guard = _first_filter(_named_rule(plan, "PyReasonRule0_copy_unknown"))
  assert isinstance(guard.expression, ScalarOr)
  crossed = guard.expression.arguments[-1]
  assert isinstance(crossed, ScalarCompare)
  assert crossed.operator == ">"


def test_disabling_inconsistency_check_is_rejected_not_silently_changed(
  tmp_path: Path,
) -> None:
  source = SourceProgram(
    rules=(_rule("copy", "output", "input", body_lower=1.0, body_upper=1.0),),
    facts=(_fact("input", ("a",), 1.0, 1.0),),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(),
    settings=(("inconsistency_check", False),),
  )

  with pytest.raises(
    UnsupportedAnnotationSemantics,
    match="inconsistency_check=False",
  ):
    compile_source(source, timesteps=0, output_root=tmp_path)


def test_nonpersistent_label_presence_resets_to_bottom_at_next_time(
  tmp_path: Path,
) -> None:
  source = SourceProgram(
    rules=(_rule("copy-unknown", "q", "p", body_upper=1.0),),
    facts=(
      SourceFact("", "p-now", "p", ("a",), 1.0, 1.0, 0, 0, False),
      SourceFact("", "keep-live", "tick", ("z",), 1.0, 1.0, 1, 1, False),
    ),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(),
    node_domain=("a", "z"),
  )

  plan = compile_source(source, timesteps=1, output_root=tmp_path)

  assert any(rule.name == "ResetLabelp" for rule in plan.program.rules)
  assert (plan.data_dir / "tick_1.csv").read_text().splitlines() == ["0,1"]


def test_conflict_is_canonicalized_and_frozen_without_a_row_id(
  tmp_path: Path,
) -> None:
  source = SourceProgram(
    rules=(_rule("observe-unknown", "q", "p", body_upper=1.0),),
    facts=(
      SourceFact("", "left", "p", ("a",), 0.8, 1.0, 0, 0, False),
      SourceFact("", "right", "p", ("a",), 0.0, 0.2, 0, 0, False),
      SourceFact("", "keep-live", "tick", ("z",), 1.0, 1.0, 1, 1, False),
    ),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(),
    node_domain=("a", "z"),
  )

  plan = compile_source(source, timesteps=1, output_root=tmp_path)

  names = {rule.name for rule in plan.program.rules}
  assert {"CanonicalConflictp", "FreezeConflictp"} <= names


def test_empty_static_fact_schedule_has_no_seed_or_static_collision(
  tmp_path: Path,
) -> None:
  source = SourceProgram(
    rules=(_rule("derive-p", "p", "base", body_lower=1.0, body_upper=1.0),),
    facts=(
      SourceFact("", "empty", "p", ("a",), 0.8, 1.0, 1, 0, True),
      SourceFact("", "base", "base", ("a",), 1.0, 1.0, 0, 0, False),
    ),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(),
    node_domain=("a",),
  )

  plan = compile_source(source, timesteps=1, output_root=tmp_path)
  output = next(item for item in plan.outputs if item.predicate == "p")
  index = int(output.relation.removeprefix("PyReasonRelation"))
  assert (plan.data_dir / f"seed_{index}.csv").read_text() == ""


def test_repeated_binary_variables_reject_pyreason_oracle_quirk(
  tmp_path: Path,
) -> None:
  repeated = SourceRule(
    text="q(X) <- p(X,X)",
    name="repeated",
    head_predicate="q",
    head_terms=("X",),
    head_annotation=None,
    head_lower=1.0,
    head_upper=1.0,
    delay=0,
    clauses=(SourceClause("p", ("X", "X"), 0.0, 0.0),),
  )
  source = SourceProgram(
    rules=(repeated,),
    facts=(),
    graphml_path=None,
    closed_world_predicates=frozenset({"p"}),
    annotation_functions=(),
    node_domain=("a", "b"),
    edge_domain=(("a", "b"),),
  )

  with pytest.raises(UnsupportedAnnotationSemantics, match="repeated variables"):
    compile_source(source, timesteps=0, output_root=tmp_path)


def test_perfect_convergence_separates_update_clock_from_live_horizon(
  tmp_path: Path,
) -> None:
  source = SourceProgram(
    rules=(_rule("default-read", "q", "p"),),
    facts=(_fact("domain", ("a",), 1.0, 1.0),),
    graphml_path=None,
    closed_world_predicates=frozenset({"p"}),
    annotation_functions=(),
    node_domain=("a",),
  )

  plan = compile_source(source, timesteps=3, output_root=tmp_path)

  assert plan.live_time_relation == "PyReasonLiveTime"
  assert (plan.data_dir / "horizon_event.csv").read_text().splitlines() == ["0"]
  assert (plan.data_dir / "update_clock.csv").read_text().splitlines() == ["0"]
  assert (plan.data_dir / "round_sync.csv").read_text().splitlines() == [
    "0",
    "1",
    "2",
    "3",
  ]
  compiled = _named_rule(plan, "PyReasonRule0_default_read")
  assert any(
    isinstance(clause, Atom) and clause.rel == "PyReasonUpdateClock" for clause in compiled.body
  )
  reset = _named_rule(plan, "ResetLabelq")
  assert any(isinstance(clause, Atom) and clause.rel == "PyReasonLiveTime" for clause in reset.body)


def test_future_fact_beyond_requested_bound_keeps_requested_time_live(
  tmp_path: Path,
) -> None:
  source = SourceProgram(
    rules=(),
    facts=(SourceFact("", "future", "p", ("a",), 1, 1, 10, 10, False),),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(),
    node_domain=("a",),
  )

  plan = compile_source(source, timesteps=5, output_root=tmp_path)

  assert (plan.data_dir / "horizon_event.csv").read_text().splitlines() == ["5"]
  assert (plan.data_dir / "update_clock.csv").read_text() == ""
  output = next(item for item in plan.outputs if item.predicate == "p")
  index = int(output.relation.removeprefix("PyReasonRelation"))
  assert (plan.data_dir / f"seed_{index}.csv").read_text() == ""


def test_delayed_bottom_candidate_extends_horizon_but_not_update_clock(
  tmp_path: Path,
) -> None:
  delayed = replace(
    _rule("delayed-bottom", "out", "trigger", body_lower=1.0, body_upper=1.0),
    delay=2,
    head_lower=0.0,
    head_upper=1.0,
  )
  source = SourceProgram(
    rules=(delayed,),
    facts=(_fact("trigger", ("a",), 1.0, 1.0),),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(),
    node_domain=("a",),
  )

  plan = compile_source(source, timesteps=4, output_root=tmp_path)

  names = {rule.name for rule in plan.program.rules}
  assert {
    "PyReasonRule0_delayed_bottom__horizon_event",
    "PyReasonRule0_delayed_bottom__update_event",
  } <= names
  activation = _named_rule(plan, "PyReasonRule0_delayed_bottom__update_event")
  assert any(isinstance(clause, Filter) for clause in activation.body)


def test_bottom_key_admission_cannot_activate_a_downstream_source_round(
  tmp_path: Path,
) -> None:
  admit_bottom = replace(
    _rule("admit-bottom", "p", "trigger", body_lower=0.0, body_upper=1.0),
    head_lower=0.0,
    head_upper=1.0,
  )
  consume_bottom = _rule(
    "consume-bottom",
    "q",
    "p",
    body_lower=0.0,
    body_upper=1.0,
  )
  source = SourceProgram(
    rules=(admit_bottom, consume_bottom),
    facts=(_fact("trigger", ("a",), 1.0, 1.0),),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(),
    node_domain=("a",),
  )

  with pytest.raises(
    UnsupportedAnnotationSemantics,
    match="logical head-change generation",
  ):
    compile_source(source, timesteps=0, output_root=tmp_path)


def test_persistent_delayed_activation_is_rejected_without_strict_change_event(
  tmp_path: Path,
) -> None:
  source = SourceProgram(
    rules=(
      replace(
        _rule("delayed", "out", "trigger", body_lower=1.0, body_upper=1.0),
        delay=1,
      ),
    ),
    facts=(_fact("trigger", ("a",), 1.0, 1.0),),
    graphml_path=None,
    closed_world_predicates=frozenset(),
    annotation_functions=(),
    node_domain=("a",),
    settings=(("persistent", True),),
  )

  with pytest.raises(UnsupportedAnnotationSemantics, match="strict-change event"):
    compile_source(source, timesteps=3, output_root=tmp_path)
