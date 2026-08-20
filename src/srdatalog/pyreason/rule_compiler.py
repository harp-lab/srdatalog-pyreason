"""Unified lowering of one captured PyReason rule into SRDatalog Core.

Rules with constant heads and rules with callbacks share this target lowering.
Python AST nodes and the private objects below exist only while interpreting a
callback's control flow. The sole output is an SRDatalog Rule; there is no
PyReason-specific annotation IR between the two languages.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal

from srdatalog import float32_to_u32, max_lower_lattice
from srdatalog.dsl import (
  Filter,
  Let,
  ScalarAnd,
  ScalarCompare,
  ScalarConst,
  ScalarExpr,
  ScalarMax,
  ScalarMin,
  ScalarOr,
  ScalarVar,
)


class RuleCompileError(NotImplementedError):
  """A source rule uses behavior outside the compilable fragment."""


@dataclass(frozen=True)
class CompiledSourceRule:
  '''Materialization plus optional demand-instrumented witness rules.'''

  materialization: Any
  preparation: tuple[Any, ...] = ()
  temporal_events: tuple[Any, ...] = ()
  witnesses: tuple[Any, ...] = ()
  diagnostic_relations: tuple[str, ...] = ()


@dataclass(frozen=True)
class WitnessRelations:
  '''Compiler-owned relations for one demanded aggregate occurrence.

  ``candidate`` retains source-body groundings, ``winner`` performs the same
  grouped selection as the source head, and ``history`` freezes every selected
  winner version under ordinary set semantics.  None is a physical tuple-ID.
  '''

  candidate: Any
  winner: Any
  history: Any


@dataclass(frozen=True)
class _Family:
  kind: Literal["annotations", "qualified_nodes", "qualified_edges"]


@dataclass(frozen=True)
class _ClauseRows:
  kind: Literal["annotations", "qualified_nodes", "qualified_edges"]
  clause: int


@dataclass(frozen=True)
class _RowIndex:
  kind: Literal["annotations", "qualified_nodes", "qualified_edges"]
  clause: int
  variable: str


@dataclass(frozen=True)
class _AnnotationRow:
  clause: int
  variable: str


@dataclass(frozen=True)
class _TermRow:
  clause: int
  variable: str


@dataclass(frozen=True)
class _TermColumn:
  clause: int
  column: int
  variable: str


@dataclass(frozen=True)
class _DynamicRange:
  rows: _ClauseRows


@dataclass(frozen=True)
class _Comparison:
  left: Any
  operator: type[ast.cmpop]
  right: Any


@dataclass(frozen=True)
class _Predicate:
  comparisons: tuple[_Comparison, ...]


@dataclass(frozen=True)
class _Loop:
  rows: _ClauseRows
  variable: str


@dataclass
class _LoopState:
  assigned: set[str]
  written: set[str]


@dataclass(frozen=True)
class _Selection:
  rank_clause: int
  lower: ScalarExpr
  upper: ScalarExpr
  initial_lower: ScalarConst
  initial_upper: ScalarConst
  prefer_first_candidate: bool


class _BreakFlow(Exception):
  pass


class _ContinueFlow(Exception):
  pass


class _ReturnFlow(Exception):
  def __init__(self, value: Any):
    self.value = value


def callback_source(function: object) -> str:
  """Return stable source text for compilation and cache fingerprinting."""
  python_function = getattr(function, "py_func", function)
  if not callable(python_function):
    raise RuleCompileError("registered annotation is not callable")
  try:
    return textwrap.dedent(inspect.getsource(python_function))
  except (OSError, TypeError) as exc:
    raise RuleCompileError(
      f"cannot inspect annotation {getattr(python_function, '__name__', function)!r}"
    ) from exc


def compile_rule(
  python_source: str | None,
  rule: Any,
  rule_index: int,
  logical: dict[str, Any],
  admission_ranks: dict[str, Any],
  ticks: dict[int, Any],
  head_predicates: set[str],
  target_name: str,
  round_sync: Any,
  update_clock: Any,
  horizon_event: Any,
  *,
  closed_world_predicates: frozenset[str] = frozenset(),
  closed_world_views: dict[str, Any] | None = None,
  closed_world_eligibility: dict[str, Any] | None = None,
  head_edge_domain: Any | None = None,
  witness_relations: WitnessRelations | None = None,
) -> CompiledSourceRule:
  """Compile one source rule directly into an SRDatalog Rule."""
  read_views = {} if closed_world_views is None else closed_world_views
  eligibility = {} if closed_world_eligibility is None else closed_world_eligibility
  if python_source is None:
    return _compile_constant_rule(
      rule,
      rule_index,
      target_name,
      logical,
      ticks,
      round_sync,
      update_clock,
      horizon_event,
      closed_world_predicates,
      read_views,
      eligibility,
      head_edge_domain,
      witness_relations,
    )
  return _compile_callback_rule(
    python_source,
    rule,
    rule_index,
    logical,
    admission_ranks,
    ticks,
    head_predicates,
    target_name,
    round_sync,
    update_clock,
    horizon_event,
    closed_world_predicates,
    read_views,
    eligibility,
    head_edge_domain,
    witness_relations,
  )


def _compile_callback_rule(
  python_source: str,
  rule: Any,
  rule_index: int,
  logical: dict[str, Any],
  admission_ranks: dict[str, Any],
  ticks: dict[int, Any],
  head_predicates: set[str],
  target_name: str,
  round_sync: Any,
  update_clock: Any,
  horizon_event: Any,
  closed_world_predicates: frozenset[str],
  closed_world_views: dict[str, Any],
  closed_world_eligibility: dict[str, Any],
  head_edge_domain: Any | None,
  witness_relations: WitnessRelations | None,
) -> CompiledSourceRule:
  from srdatalog import Var

  if witness_relations is not None:
    raise RuleCompileError(
      "callback provenance requires a logical aggregate-change event and "
      "same-snapshot group membership"
    )

  tree = ast.parse(python_source)
  definition = next(
    (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
    None,
  )
  if definition is None or isinstance(definition, ast.AsyncFunctionDef):
    raise RuleCompileError("annotation must be a synchronous Python function")
  positional = [*definition.args.posonlyargs, *definition.args.args]
  if (
    definition.args.vararg is not None
    or definition.args.kwarg is not None
    or definition.args.kwonlyargs
    or len(positional) not in {2, 6}
  ):
    raise RuleCompileError("annotation must use PyReason's two- or six-argument signature")
  if len(positional) == 2:
    raise RuleCompileError(
      "two-argument scalar/group callbacks are not yet in the compilable fragment"
    )

  term_vars = _term_variables(rule, rule_index)
  body_time = Var(f"rule_{rule_index}_time")
  lowers = [Var(f"rule_{rule_index}_lower_{i}") for i in range(len(rule.clauses))]
  uppers = [Var(f"rule_{rule_index}_upper_{i}") for i in range(len(rule.clauses))]
  compiler = _Compiler(rule, lowers, uppers)
  names = [argument.arg for argument in positional]
  environment: dict[str, Any] = {
    names[0]: _Family("annotations"),
    names[1]: tuple(rule.weights),
    names[2]: _Family("qualified_nodes"),
    names[3]: _Family("qualified_edges"),
    names[4]: tuple(_clause_predicate(clause) for clause in rule.clauses),
    names[5]: tuple(_clause_terms(clause) for clause in rule.clauses),
  }
  with suppress(_ReturnFlow):
    compiler.execute(definition.body, environment, ())
  return compiler.finish_rule(
    rule_index=rule_index,
    logical=logical,
    admission_ranks=admission_ranks,
    ticks=ticks,
    term_vars=term_vars,
    body_time=body_time,
    lowers=lowers,
    uppers=uppers,
    head_predicates=head_predicates,
    target_name=target_name,
    round_sync=round_sync,
    update_clock=update_clock,
    horizon_event=horizon_event,
    closed_world_predicates=closed_world_predicates,
    closed_world_views=closed_world_views,
    closed_world_eligibility=closed_world_eligibility,
    head_edge_domain=head_edge_domain,
    witness_relations=witness_relations,
  )


class _Compiler:
  def __init__(self, rule: Any, lowers: list[Any], uppers: list[Any]):
    self.rule = rule
    self.lowers = lowers
    self.uppers = uppers
    self.selection: _Selection | None = None
    self.final_return: tuple[ScalarExpr, ScalarExpr] | None = None
    self.first_candidate_flags: set[str] = set()
    self.empty_group_fallback: tuple[ScalarConst, ScalarConst] | None = None
    self.drop_crossed_interval = False
    self.dynamic_loop_clauses: set[int] = set()
    self.lookup_bound_names: set[str] = set()
    self.protected_selection_names: set[str] = set()
    self.selection_assignment_depth = 0
    self.escaped_dynamic_names: set[str] = set()
    self.definition_loop_depth: dict[str, int] = {}
    self.dynamic_loop_states: list[_LoopState] = []
    self.condition_loop_reads: set[str] = set()

  def fail(self, node: ast.AST, message: str) -> RuleCompileError:
    line = getattr(node, "lineno", "?")
    return RuleCompileError(f"annotation line {line}: {message}")

  def execute(
    self,
    statements: list[ast.stmt],
    environment: dict[str, Any],
    loops: tuple[_Loop, ...],
    *,
    direct_dynamic_body: bool = False,
  ) -> None:
    for statement_index, statement in enumerate(statements):
      if isinstance(statement, ast.Assign):
        if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
          raise self.fail(statement, "only simple local assignments are supported")
        target = statement.targets[0].id
        if target in self.protected_selection_names and self.selection_assignment_depth == 0:
          raise self.fail(
            statement,
            f"selected accumulator {target!r} cannot be mutated after selection",
          )
        environment[target] = self.expression(statement.value, environment)
        self._record_loop_write(target)
        self.definition_loop_depth[target] = len(loops)
        self.escaped_dynamic_names.discard(target)
        continue
      if isinstance(statement, ast.AnnAssign):
        if not isinstance(statement.target, ast.Name) or statement.value is None:
          raise self.fail(statement, "only initialized local annotations are supported")
        target = statement.target.id
        if target in self.protected_selection_names and self.selection_assignment_depth == 0:
          raise self.fail(
            statement,
            f"selected accumulator {target!r} cannot be mutated after selection",
          )
        environment[target] = self.expression(statement.value, environment)
        self._record_loop_write(target)
        self.definition_loop_depth[target] = len(loops)
        self.escaped_dynamic_names.discard(target)
        continue
      if isinstance(statement, ast.For):
        if statement.orelse:
          raise self.fail(statement, "callback for-else control flow is unsupported")
        if not isinstance(statement.target, ast.Name):
          raise self.fail(statement, "loop targets must be local names")
        iterable = self.expression(statement.iter, environment)
        if isinstance(iterable, range):
          for item in iterable:
            environment[statement.target.id] = item
            self._record_loop_write(statement.target.id)
            self.definition_loop_depth[statement.target.id] = len(loops)
            try:
              self.execute(statement.body, environment, loops)
            except _ContinueFlow:
              continue
            except _BreakFlow:
              break
          continue
        if isinstance(iterable, _DynamicRange):
          if iterable.rows.clause in self.dynamic_loop_clauses:
            raise self.fail(
              statement,
              "multiple dynamic loops over one callback clause require an "
              "explicit self-join occurrence",
            )
          self.dynamic_loop_clauses.add(iterable.rows.clause)
          before = dict(environment)
          state = _LoopState(
            assigned=self._assigned_names(statement.body),
            written={statement.target.id},
          )
          self.dynamic_loop_states.append(state)
          environment[statement.target.id] = _RowIndex(
            iterable.rows.kind,
            iterable.rows.clause,
            statement.target.id,
          )
          self.definition_loop_depth[statement.target.id] = len(loops) + 1
          self._record_loop_write(statement.target.id)
          self.escaped_dynamic_names.discard(statement.target.id)
          try:
            with suppress(_ContinueFlow, _BreakFlow):
              self.execute(
                statement.body,
                environment,
                (*loops, _Loop(iterable.rows, statement.target.id)),
                direct_dynamic_body=True,
              )
          finally:
            self.dynamic_loop_states.pop()
          self._close_dynamic_scope(
            before,
            environment,
            outermost=not loops,
          )
          continue
        raise self.fail(statement, "loop iterable is not a clause collection")
      if isinstance(statement, ast.If):
        permitted_reads = self._condition_state_names(statement.test)
        self.condition_loop_reads.update(permitted_reads)
        try:
          condition = self.expression(statement.test, environment)
        finally:
          self.condition_loop_reads.difference_update(permitted_reads)
        if isinstance(condition, bool):
          if not condition and self._capture_empty_group_fallback(
            statement,
            environment,
          ):
            continue
          branch = statement.body if condition else statement.orelse
          self.execute(branch, environment, loops)
          continue
        if self._selection_if(statement, environment, loops):
          if not direct_dynamic_body:
            raise self.fail(
              statement,
              "aggregate selection must occur directly in its dynamic loop body",
            )
          if statement_index != len(statements) - 1:
            raise self.fail(
              statement,
              "aggregate selection must be the final statement in its dynamic loop body",
            )
          self.selection_assignment_depth += 1
          try:
            self.execute(statement.body, environment, loops)
          finally:
            self.selection_assignment_depth -= 1
          continue
        if self._capture_crossed_fallback(statement, environment, condition):
          continue
        if self._is_impossible_negative_guard(statement, condition):
          continue
        if self._merge_lookup_conditional(
          statement,
          environment,
          condition,
          loops,
        ):
          if not direct_dynamic_body:
            raise self.fail(
              statement,
              "aligned lookup must occur directly in its dynamic loop body",
            )
          if statement_index != len(statements) - 1:
            raise self.fail(
              statement,
              "aligned lookup must be the final statement in its dynamic loop body",
            )
          raise _BreakFlow
        raise self.fail(
          statement,
          "dynamic conditional is outside the proven lookup/aggregate fragment",
        )
      if isinstance(statement, ast.Return):
        value = self.expression(statement.value, environment)
        if self._is_interval_expression(value):
          self.final_return = value
        raise _ReturnFlow(value)
      if isinstance(statement, ast.Break):
        raise _BreakFlow
      if isinstance(statement, ast.Continue):
        raise _ContinueFlow
      if isinstance(statement, ast.Expr):
        self.expression(statement.value, environment)
        continue
      raise self.fail(statement, f"unsupported statement {type(statement).__name__}")

  def expression(self, node: ast.AST | None, environment: dict[str, Any]) -> Any:
    if node is None:
      return None
    if isinstance(node, ast.Constant):
      return node.value
    if isinstance(node, ast.Name):
      for state in reversed(self.dynamic_loop_states):
        if (
          node.id in state.assigned
          and node.id not in state.written
          and node.id not in self.condition_loop_reads
        ):
          raise self.fail(
            node,
            f"local {node.id!r} carries unrecognized state between dynamic loop iterations",
          )
      if node.id in self.escaped_dynamic_names:
        raise self.fail(
          node,
          f"row-derived local {node.id!r} escapes its dynamic loop",
        )
      if node.id not in environment:
        raise self.fail(node, f"unknown local {node.id!r}")
      return environment[node.id]
    if isinstance(node, ast.Tuple):
      return tuple(self.expression(item, environment) for item in node.elts)
    if isinstance(node, ast.List):
      return [self.expression(item, environment) for item in node.elts]
    if isinstance(node, ast.Attribute):
      value = self.expression(node.value, environment)
      if isinstance(value, _AnnotationRow) and node.attr in {"lower", "upper"}:
        variables = self.lowers if node.attr == "lower" else self.uppers
        return ScalarVar(variables[value.clause].name)
      if isinstance(value, str) and node.attr == "value":
        return value
      raise self.fail(node, f"unsupported attribute .{node.attr}")
    if isinstance(node, ast.Subscript):
      value = self.expression(node.value, environment)
      index = self.expression(node.slice, environment)
      if isinstance(value, _Family):
        clause = self._clause_index(node, index)
        return _ClauseRows(value.kind, clause)
      if isinstance(value, _ClauseRows):
        if not isinstance(index, _RowIndex) or index.clause != value.clause:
          raise self.fail(node, "clause rows must use their aligned loop index")
        if value.kind == "annotations":
          return _AnnotationRow(value.clause, index.variable)
        if value.kind == "qualified_nodes":
          return _TermColumn(value.clause, 0, index.variable)
        return _TermRow(value.clause, index.variable)
      if isinstance(value, _TermRow):
        column = self._static_int(node, index)
        terms = _clause_terms(self.rule.clauses[value.clause])
        if not 0 <= column < len(terms):
          raise self.fail(node, "qualified edge column is out of range")
        return _TermColumn(value.clause, column, value.variable)
      try:
        return value[index]
      except (IndexError, KeyError, TypeError) as exc:
        raise self.fail(node, "invalid static subscript") from exc
    if isinstance(node, ast.Call):
      if not isinstance(node.func, ast.Name):
        raise self.fail(node, "only direct calls to len/range/min/max are supported")
      if node.keywords:
        raise self.fail(node, "len/range/min/max keyword arguments are unsupported")
      arguments = [self.expression(argument, environment) for argument in node.args]
      if node.func.id == "len" and len(arguments) == 1:
        value = arguments[0]
        if isinstance(value, _ClauseRows):
          return _DynamicRange(value)
        return len(value)
      if node.func.id == "range" and len(arguments) == 1:
        value = arguments[0]
        return value if isinstance(value, _DynamicRange) else range(value)
      if node.func.id in {"min", "max"} and arguments:
        flattened = tuple(self._scalar_expression(node, value) for value in arguments)
        function: Literal["min", "max"] = "min" if node.func.id == "min" else "max"
        return self._simplify_call(function, flattened)
      raise self.fail(node, f"unsupported call {node.func.id}(...)")
    if isinstance(node, ast.UnaryOp):
      operand = self.expression(node.operand, environment)
      if isinstance(node.op, ast.Not):
        if isinstance(operand, bool):
          return not operand
        if isinstance(operand, _Predicate):
          return operand
      if isinstance(node.op, ast.USub) and isinstance(operand, (int, float)):
        return -operand
      raise self.fail(node, "unsupported unary expression")
    if isinstance(node, ast.BoolOp):
      values = [self.expression(value, environment) for value in node.values]
      if all(isinstance(value, bool) for value in values):
        return all(values) if isinstance(node.op, ast.And) else any(values)
      predicate_comparisons = tuple(
        comparison
        for value in values
        if isinstance(value, _Predicate)
        for comparison in value.comparisons
      )
      return _Predicate(predicate_comparisons)
    if isinstance(node, ast.Compare):
      left = self.expression(node.left, environment)
      comparison_items: list[_Comparison] = []
      static = True
      static_result = True
      for operator, comparator_node in zip(node.ops, node.comparators):
        right = self.expression(comparator_node, environment)
        if self._is_static(left) and self._is_static(right):
          static_result = static_result and self._compare(left, operator, right)
        else:
          static = False
          comparison_items.append(_Comparison(left, type(operator), right))
        left = right
      return static_result if static else _Predicate(tuple(comparison_items))
    raise self.fail(node, f"unsupported expression {type(node).__name__}")

  def _selection_if(
    self,
    statement: ast.If,
    environment: dict[str, Any],
    loops: tuple[_Loop, ...],
  ) -> bool:
    guard = self._selection_guard(statement.test)
    if guard is None:
      return False
    comparison, first_candidate_flag = guard
    assignments = {
      item.targets[0].id: item.value.id
      for item in statement.body
      if isinstance(item, ast.Assign)
      and len(item.targets) == 1
      and isinstance(item.targets[0], ast.Name)
      and isinstance(item.value, ast.Name)
    }
    if not isinstance(comparison.left, ast.Name) or not isinstance(
      comparison.comparators[0], ast.Name
    ):
      return False
    candidate = comparison.left.id
    accumulator = comparison.comparators[0].id
    if assignments.get(accumulator) != candidate:
      return False
    if statement.orelse:
      raise self.fail(statement, "aggregate selection cannot have an else branch")
    if not isinstance(comparison.ops[0], ast.Gt):
      raise self.fail(
        comparison,
        "grouped-head compilation currently requires strict maximum selection",
      )
    if not loops:
      raise self.fail(statement, "aggregate selection is not inside a clause loop")
    lower = self._scalar_expression(statement, environment[candidate])
    upper_candidates = [
      (target, source)
      for target, source in assignments.items()
      if target != accumulator
      and target in environment
      and source in environment
      and isinstance(environment[source], ScalarExpr)
    ]
    if len(upper_candidates) != 1:
      raise self.fail(statement, "cannot identify the selected upper-bound accumulator")
    upper_accumulator, upper_candidate = upper_candidates[0]
    upper = self._scalar_expression(statement, environment[upper_candidate])
    initial_lower = self._scalar_expression(statement, environment[accumulator])
    initial_upper = self._scalar_expression(statement, environment[upper_accumulator])
    if not isinstance(initial_lower, ScalarConst) or not isinstance(initial_upper, ScalarConst):
      raise self.fail(
        statement,
        "aggregate accumulator initializers must be static interval endpoints",
      )
    body_targets: list[str] = []
    true_assignments: set[str] = set()
    for item in statement.body:
      if not (
        isinstance(item, ast.Assign)
        and len(item.targets) == 1
        and isinstance(item.targets[0], ast.Name)
      ):
        raise self.fail(
          item,
          "aggregate selection body must contain only simple assignments",
        )
      target = item.targets[0].id
      body_targets.append(target)
      if isinstance(item.value, ast.Constant) and item.value.value is True:
        true_assignments.add(target)
    prefer_first_candidate = first_candidate_flag is not None
    expected_targets = {accumulator, upper_accumulator}
    initializer_names = {accumulator, upper_accumulator}
    if prefer_first_candidate:
      assert first_candidate_flag is not None
      if environment.get(first_candidate_flag) is not False or true_assignments != {
        first_candidate_flag
      }:
        raise self.fail(
          statement,
          "first-candidate selection must change its false guard to true",
        )
      expected_targets.add(first_candidate_flag)
      initializer_names.add(first_candidate_flag)
      self.first_candidate_flags.add(first_candidate_flag)
    elif true_assignments:
      raise self.fail(
        statement,
        "plain maximum selection cannot mutate an unrelated boolean flag",
      )
    if len(body_targets) != len(expected_targets) or set(body_targets) != expected_targets:
      raise self.fail(
        statement,
        "aggregate selection body may update only its lower, upper, and "
        "recognized first-candidate flag",
      )
    if any(self.definition_loop_depth.get(name) != 0 for name in initializer_names):
      raise self.fail(
        statement,
        "aggregate accumulators and first-candidate flags must be initialized "
        "before the outermost dynamic loop",
      )
    self.protected_selection_names.update(expected_targets)
    selection = _Selection(
      rank_clause=loops[0].rows.clause,
      lower=lower,
      upper=upper,
      initial_lower=initial_lower,
      initial_upper=initial_upper,
      prefer_first_candidate=prefer_first_candidate,
    )
    if self.selection is not None and self.selection != selection:
      raise self.fail(statement, "multiple different aggregate selections are unsupported")
    self.selection = selection
    return True

  @staticmethod
  def _selection_guard(node: ast.AST) -> tuple[ast.Compare, str | None] | None:
    if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
      return node, None
    if not (isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or) and len(node.values) == 2):
      return None
    comparison = next(
      (
        value
        for value in node.values
        if isinstance(value, ast.Compare) and len(value.ops) == len(value.comparators) == 1
      ),
      None,
    )
    negated = next(
      (
        value
        for value in node.values
        if isinstance(value, ast.UnaryOp)
        and isinstance(value.op, ast.Not)
        and isinstance(value.operand, ast.Name)
      ),
      None,
    )
    if comparison is None or negated is None:
      return None
    assert isinstance(negated.operand, ast.Name)
    return comparison, negated.operand.id

  def _capture_empty_group_fallback(
    self,
    statement: ast.If,
    environment: dict[str, Any],
  ) -> bool:
    if not (
      isinstance(statement.test, ast.UnaryOp)
      and isinstance(statement.test.op, ast.Not)
      and isinstance(statement.test.operand, ast.Name)
      and statement.test.operand.id in self.first_candidate_flags
      and len(statement.body) == 1
      and isinstance(statement.body[0], ast.Return)
      and not statement.orelse
    ):
      return False
    value = self.expression(statement.body[0].value, environment)
    if not (
      isinstance(value, (tuple, list))
      and len(value) == 2
      and all(isinstance(item, (int, float)) for item in value)
    ):
      raise self.fail(
        statement,
        "empty aggregate fallback must return two static interval endpoints",
      )
    self.empty_group_fallback = (
      ScalarConst(float32_to_u32(float(value[0]))),
      ScalarConst(float32_to_u32(float(value[1]))),
    )
    return True

  def _close_dynamic_scope(
    self,
    before: dict[str, Any],
    environment: dict[str, Any],
    *,
    outermost: bool,
  ) -> None:
    allowed = set(self.protected_selection_names)
    if not outermost:
      allowed.update(self.lookup_bound_names)
    for name, value in tuple(environment.items()):
      if name in allowed:
        continue
      if name not in before:
        del environment[name]
        self.escaped_dynamic_names.add(name)
        continue
      if value != before[name]:
        environment[name] = before[name]
        self.escaped_dynamic_names.add(name)

  def _record_loop_write(self, name: str) -> None:
    for state in self.dynamic_loop_states:
      if name in state.assigned:
        state.written.add(name)

  @staticmethod
  def _assigned_names(statements: list[ast.stmt]) -> set[str]:
    result: set[str] = set()

    def visit(items: list[ast.stmt]) -> None:
      for statement in items:
        if (
          isinstance(statement, ast.Assign)
          and len(statement.targets) == 1
          and isinstance(statement.targets[0], ast.Name)
        ):
          result.add(statement.targets[0].id)
          continue
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
          result.add(statement.target.id)
          continue
        if isinstance(statement, ast.If):
          visit(statement.body)
          visit(statement.orelse)
          continue
        # A nested dynamic loop owns its assignments. Treating them as writes
        # of the enclosing loop makes a valid outer aggregate accumulator look
        # like arbitrary loop-carried state.
        if isinstance(statement, ast.For):
          continue

    visit(statements)
    return result

  @staticmethod
  def _condition_state_names(node: ast.AST) -> set[str]:
    """Allow carried locals only while parsing a conditional guard.

    A recognized aggregate must compare a per-row candidate with an
    accumulator (and may test a first-candidate flag). The allowance is scoped
    to the guard expression: the same accumulator remains illegal in candidate
    computation or any other dynamic-loop statement.
    """
    result: set[str] = set()
    for candidate in ast.walk(node):
      if (
        isinstance(candidate, ast.Compare)
        and len(candidate.ops) == len(candidate.comparators) == 1
        and isinstance(candidate.left, ast.Name)
        and isinstance(candidate.comparators[0], ast.Name)
      ):
        result.add(candidate.comparators[0].id)
      elif (
        isinstance(candidate, ast.UnaryOp)
        and isinstance(candidate.op, ast.Not)
        and isinstance(candidate.operand, ast.Name)
      ):
        result.add(candidate.operand.id)
    return result

  @staticmethod
  def _active_term_occurrence(
    term: _TermColumn,
    loops: tuple[_Loop, ...],
  ) -> bool:
    return any(term.clause == loop.rows.clause and term.variable == loop.variable for loop in loops)

  def _validate_lookup_occurrences(
    self,
    statement: ast.If,
    left: _TermColumn,
    right: _TermColumn,
    loops: tuple[_Loop, ...],
  ) -> None:
    current = loops[-1]
    left_is_current = left.clause == current.rows.clause and left.variable == current.variable
    right_is_current = right.clause == current.rows.clause and right.variable == current.variable
    if left_is_current == right_is_current:
      raise self.fail(
        statement,
        "lookup equality must compare the innermost row with one enclosing row",
      )
    enclosing = right if left_is_current else left
    if not self._active_term_occurrence(enclosing, loops[:-1]):
      raise self.fail(
        statement,
        "lookup equality must compare the innermost row with one enclosing row",
      )

  def _merge_lookup_conditional(
    self,
    statement: ast.If,
    environment: dict[str, Any],
    condition: Any,
    loops: tuple[_Loop, ...],
  ) -> bool:
    if not (
      loops
      and not statement.orelse
      and len(statement.body) >= 2
      and isinstance(statement.body[-1], ast.Break)
      and all(
        isinstance(item, ast.Assign)
        and len(item.targets) == 1
        and isinstance(item.targets[0], ast.Name)
        for item in statement.body[:-1]
      )
      and isinstance(statement.test, ast.Compare)
      and len(statement.test.ops) == len(statement.test.comparators) == 1
      and isinstance(statement.test.ops[0], ast.Eq)
      and isinstance(condition, _Predicate)
      and len(condition.comparisons) == 1
    ):
      return False
    comparison = condition.comparisons[0]
    if not (
      comparison.operator is ast.Eq
      and isinstance(comparison.left, _TermColumn)
      and isinstance(comparison.right, _TermColumn)
    ):
      return False
    if len(loops) < 2:
      raise self.fail(
        statement,
        "aligned lookup requires an enclosing driver row",
      )
    current = loops[-1]
    self._validate_lookup_occurrences(
      statement,
      comparison.left,
      comparison.right,
      loops,
    )
    self._validate_term_equalities(statement, condition)
    branch = dict(environment)
    initial_depths = dict(self.definition_loop_depth)
    self.execute(statement.body[:-1], branch, loops)
    changed: set[str] = set()
    allowed_values = {
      ScalarVar(self.lowers[current.rows.clause].name),
      ScalarVar(self.uppers[current.rows.clause].name),
    }
    for name, branch_value in branch.items():
      original = environment.get(name)
      if original == branch_value:
        continue
      changed.add(name)
      if isinstance(original, (int, float)) and original < 0 and branch_value in allowed_values:
        environment[name] = branch_value
        continue
      raise self.fail(
        statement,
        f"lookup assignment to {name!r} is outside the aligned-row fragment",
      )
    assignment_targets: set[str] = set()
    for item in statement.body[:-1]:
      assert isinstance(item, ast.Assign)
      assert isinstance(item.targets[0], ast.Name)
      assignment_targets.add(item.targets[0].id)
    if not changed or changed != assignment_targets:
      raise self.fail(
        statement,
        "lookup branch must replace only negative sentinels with aligned interval endpoints",
      )
    if any(initial_depths.get(name) != len(loops) - 1 for name in changed):
      raise self.fail(
        statement,
        "lookup sentinels must be reset in the immediately enclosing driver loop",
      )
    self.lookup_bound_names.update(changed)
    return True

  def _is_impossible_negative_guard(self, statement: ast.If, condition: Any) -> bool:
    if not (
      not statement.orelse
      and len(statement.body) == 1
      and isinstance(statement.body[0], ast.Continue)
      and isinstance(statement.test, ast.Compare)
      and len(statement.test.ops) == len(statement.test.comparators) == 1
      and isinstance(statement.test.ops[0], ast.Lt)
      and isinstance(statement.test.left, ast.Name)
      and statement.test.left.id in self.lookup_bound_names
      and isinstance(statement.test.comparators[0], ast.Constant)
      and type(statement.test.comparators[0].value) in {int, float}
      and statement.test.comparators[0].value == 0.0
      and isinstance(condition, _Predicate)
      and len(condition.comparisons) == 1
    ):
      return False
    comparison = condition.comparisons[0]
    return (
      comparison.operator is ast.Lt
      and isinstance(comparison.left, ScalarVar)
      and comparison.left.name in {variable.name for variable in (*self.lowers, *self.uppers)}
      and type(comparison.right) in {int, float}
      and comparison.right == 0.0
    )

  def _capture_crossed_fallback(
    self,
    statement: ast.If,
    environment: dict[str, Any],
    condition: Any,
  ) -> bool:
    if not (
      self.selection is not None
      and not statement.orelse
      and len(statement.body) == 1
      and isinstance(statement.body[0], ast.Return)
      and isinstance(statement.test, ast.Compare)
      and len(statement.test.ops) == len(statement.test.comparators) == 1
      and isinstance(statement.test.ops[0], ast.Gt)
      and isinstance(condition, _Predicate)
      and len(condition.comparisons) == 1
    ):
      return False
    value = self.expression(statement.body[0].value, environment)
    comparison = condition.comparisons[0]
    if not self._is_unknown_interval(value):
      return False
    if not (
      comparison.operator is ast.Gt
      and comparison.left == self.selection.lower
      and comparison.right == self.selection.upper
    ):
      return False
    self.drop_crossed_interval = True
    return True

  def _validate_term_equalities(self, node: ast.AST, condition: Any) -> None:
    if not isinstance(condition, _Predicate):
      return
    for comparison in condition.comparisons:
      if comparison.operator is not ast.Eq:
        continue
      if not isinstance(comparison.left, _TermColumn) or not isinstance(
        comparison.right, _TermColumn
      ):
        continue
      left_term = _clause_terms(self.rule.clauses[comparison.left.clause])[comparison.left.column]
      right_term = _clause_terms(self.rule.clauses[comparison.right.clause])[
        comparison.right.column
      ]
      if left_term != right_term:
        raise self.fail(
          node,
          "callback adds an equality absent from the Datalog rule body",
        )

  @staticmethod
  def _is_unknown_interval(value: Any) -> bool:
    return value in {(0, 1), (0.0, 1.0)}

  @staticmethod
  def _is_interval_expression(value: Any) -> bool:
    return (
      isinstance(value, tuple)
      and len(value) == 2
      and all(isinstance(item, ScalarExpr) for item in value)
    )

  @staticmethod
  def _is_static(value: Any) -> bool:
    if isinstance(value, (tuple, list)):
      return all(_Compiler._is_static(item) for item in value)
    return isinstance(value, (str, int, float, bool, range))

  @staticmethod
  def _compare(left: Any, operator: ast.cmpop, right: Any) -> bool:
    if isinstance(operator, ast.Eq):
      return bool(left == right)
    if isinstance(operator, ast.NotEq):
      return bool(left != right)
    if isinstance(operator, ast.Lt):
      return bool(left < right)
    if isinstance(operator, ast.LtE):
      return bool(left <= right)
    if isinstance(operator, ast.Gt):
      return bool(left > right)
    if isinstance(operator, ast.GtE):
      return bool(left >= right)
    raise RuleCompileError(f"unsupported comparison {type(operator).__name__}")

  def _clause_index(self, node: ast.AST, value: Any) -> int:
    index = self._static_int(node, value)
    if not 0 <= index < len(self.rule.clauses):
      raise self.fail(node, f"clause index {index} is out of range")
    return index

  def _static_int(self, node: ast.AST, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
      raise self.fail(node, "subscript must resolve to a static integer")
    return value

  def _scalar_expression(self, node: ast.AST, value: Any) -> ScalarExpr:
    if isinstance(value, ScalarExpr):
      return value
    if isinstance(value, (int, float)):
      return ScalarConst(float32_to_u32(float(value)))
    raise self.fail(node, "expected an interval-bound expression")

  @staticmethod
  def _simplify_call(
    function: Literal["min", "max"], arguments: tuple[ScalarExpr, ...]
  ) -> ScalarExpr:
    flattened: list[ScalarExpr] = []
    for argument in arguments:
      if (function == "min" and isinstance(argument, ScalarMin)) or (
        function == "max" and isinstance(argument, ScalarMax)
      ):
        flattened.extend(argument.arguments)
      else:
        flattened.append(argument)
    one = ScalarConst(float32_to_u32(1.0))
    zero = ScalarConst(float32_to_u32(0.0))
    if function == "min" and one in flattened:
      flattened = [item for item in flattened if item != one]
    if function == "max" and zero in flattened:
      flattened = [item for item in flattened if item != zero]
    if not flattened:
      return one if function == "min" else zero
    if len(flattened) == 1:
      return flattened[0]
    expression_type = ScalarMin if function == "min" else ScalarMax
    return expression_type(tuple(flattened))

  def finish_rule(
    self,
    *,
    rule_index: int,
    logical: dict[str, Any],
    admission_ranks: dict[str, Any],
    ticks: dict[int, Any],
    term_vars: dict[str, Any],
    body_time: Any,
    lowers: list[Any],
    uppers: list[Any],
    head_predicates: set[str],
    target_name: str,
    round_sync: Any,
    update_clock: Any,
    horizon_event: Any,
    closed_world_predicates: frozenset[str],
    closed_world_views: dict[str, Any],
    closed_world_eligibility: dict[str, Any],
    head_edge_domain: Any | None,
    witness_relations: WitnessRelations | None,
  ) -> CompiledSourceRule:
    if self.selection is None:
      raise RuleCompileError("callback does not contain a supported rule-local maximum reduction")
    if self.final_return is None:
      raise RuleCompileError("callback has no compilable interval return")
    rank_clause = self.selection.rank_clause
    selected_lower = self.selection.lower
    selected_upper = self.selection.upper
    if self.final_return != (selected_lower, selected_upper):
      raise RuleCompileError(
        "callback final result is not the interval carried by its selected witness"
      )
    lower_inputs = {lower.name for lower in lowers}
    upper_inputs = {upper.name for upper in uppers}
    if not _scalar_variable_names(selected_lower) <= lower_inputs:
      raise RuleCompileError(
        "maximum-selection lower must depend only on source lower endpoints; "
        "otherwise retained interval versions are not monotone"
      )
    if not _scalar_variable_names(selected_upper) <= upper_inputs:
      raise RuleCompileError(
        "maximum-selection upper must depend only on source upper endpoints; "
        "otherwise retained interval versions are not monotone"
      )

    rank_predicate = self.rule.clauses[rank_clause].predicate
    if rank_predicate not in admission_ranks or rank_predicate in head_predicates:
      raise RuleCompileError("maximum-selection driver must be an extensional relation")
    rank_source_clause = self.rule.clauses[rank_clause]
    if rank_predicate in closed_world_predicates and rank_source_clause.lower == 0.0:
      raise RuleCompileError("maximum-selection driver cannot use a closed-world default range")

    from srdatalog import Relation, Var

    head_time = body_time
    rank = Var(f"rule_{rule_index}_rank")
    clauses: list[Any] = []
    bound_terms: set[str] = set()
    for clause_index, clause in enumerate(self.rule.clauses):
      arguments = tuple(term_vars[term] for term in clause.terms)
      if clause_index == rank_clause:
        clauses.append(
          admission_ranks[clause.predicate](
            *arguments,
            rank,
          )
        )
      relation = (
        closed_world_views.get(clause.predicate, logical[clause.predicate])
        if clause.lower == 0.0
        else logical[clause.predicate]
      )
      atom_arguments = [*arguments, body_time]
      atom_arguments.extend((lowers[clause_index], uppers[clause_index]))
      clauses.extend(
        (
          relation(*atom_arguments),
          _guard(
            lowers[clause_index],
            uppers[clause_index],
            clause.lower,
            clause.upper,
            closed_world=(clause.predicate in closed_world_predicates and clause.lower == 0.0),
          ),
        )
      )
      if (
        clause.lower == 0.0
        and clause.predicate in closed_world_eligibility
        and all(term not in bound_terms for term in clause.terms)
      ):
        clauses.append(closed_world_eligibility[clause.predicate](*arguments, body_time))
      bound_terms.update(clause.terms)
    clauses.extend((round_sync(body_time), update_clock(body_time)))
    if self.rule.delay > 0:
      head_time = Var(f"rule_{rule_index}_next_time")
      clauses.append(ticks[self.rule.delay](body_time, head_time))

    result_lower = Var(f"rule_{rule_index}_result_lower")
    result_upper = Var(f"rule_{rule_index}_result_upper")
    clauses.extend(
      (
        Let(result_lower.name, expression=self.final_return[0]),
        Let(result_upper.name, expression=self.final_return[1]),
      )
    )
    head_arguments = tuple(term_vars[term] for term in self.rule.head_terms)
    if len(head_arguments) == 2:
      if head_edge_domain is None:
        raise RuleCompileError("binary heads require an active-edge domain relation")
      clauses.append(head_edge_domain(*head_arguments, body_time))
    body = _conjunction(clauses, self.rule.name)
    proposal_arity = len(head_arguments) + 5
    driver_terms = set(rank_source_clause.terms)
    group_terms = set(self.rule.head_terms)
    witness_terms = tuple(
      term for term in term_vars if term not in driver_terms and term not in group_terms
    )
    candidate_arity = proposal_arity + len(witness_terms)
    candidate_relation = Relation(
      f"PyReasonCallback{rule_index}ProposalCandidate",
      candidate_arity,
      column_types=(int,) * candidate_arity,
    )
    proposal = Relation(
      f"PyReasonCallback{rule_index}SelectedProposal",
      proposal_arity,
      column_types=(int,) * proposal_arity,
      value_spec=max_lower_lattice(
        key_columns=tuple(range(len(head_arguments) + 2)),
        rank_column=len(head_arguments) + 2,
        lower_column=len(head_arguments) + 3,
        upper_column=len(head_arguments) + 4,
      ),
    )
    candidate_head = candidate_relation(
      *head_arguments,
      body_time,
      head_time,
      rank,
      result_lower,
      result_upper,
      *(term_vars[term] for term in witness_terms),
    )
    candidate = (candidate_head <= body).named(f"{target_name}__aggregate_candidate")
    proposal_source_rank = Var(f"rule_{rule_index}_proposal_rank")
    proposal_source_lower = Var(f"rule_{rule_index}_proposal_lower")
    proposal_source_upper = Var(f"rule_{rule_index}_proposal_upper")
    proposal_witnesses = tuple(
      Var(f"rule_{rule_index}_proposal_witness_{index}") for index in range(len(witness_terms))
    )
    proposal_projection = (
      proposal(
        *head_arguments,
        body_time,
        head_time,
        proposal_source_rank,
        proposal_source_lower,
        proposal_source_upper,
      )
      <= candidate_relation(
        *head_arguments,
        body_time,
        head_time,
        proposal_source_rank,
        proposal_source_lower,
        proposal_source_upper,
        *proposal_witnesses,
      )
    ).named(f"{target_name}__aggregate_select")

    # A real admission rank identifies one driver key, not a physical storage
    # row.  Another body clause can nevertheless introduce an existential
    # grounding that makes one driver rank produce two different callback
    # payloads. Preserve the extra logical bindings so ordinary interval-version
    # changes of one grounding remain legal, while two distinct groundings with
    # divergent payloads are diagnosed. Equal payloads are harmless under
    # Datalog set semantics.
    preparation: tuple[Any, ...] = (candidate, proposal_projection)
    diagnostic_relations: tuple[str, ...] = ()
    if witness_terms:
      violation = Relation(
        f"PyReasonCallback{rule_index}RankFunctionalViolation",
        1,
        column_types=(int,),
      )
      left_lower = Var(f"rule_{rule_index}_fd_left_lower")
      left_upper = Var(f"rule_{rule_index}_fd_left_upper")
      right_lower = Var(f"rule_{rule_index}_fd_right_lower")
      right_upper = Var(f"rule_{rule_index}_fd_right_upper")
      left_witnesses = tuple(
        Var(f"rule_{rule_index}_fd_left_witness_{index}") for index in range(len(witness_terms))
      )
      right_witnesses = tuple(
        Var(f"rule_{rule_index}_fd_right_witness_{index}") for index in range(len(witness_terms))
      )
      payload_differs = ScalarOr(
        (
          ScalarCompare(
            "!=",
            ScalarVar(left_lower.name),
            ScalarVar(right_lower.name),
          ),
          ScalarCompare(
            "!=",
            ScalarVar(left_upper.name),
            ScalarVar(right_upper.name),
          ),
        )
      )
      witness_differs = ScalarOr(
        tuple(
          ScalarCompare(
            "!=",
            ScalarVar(left.name),
            ScalarVar(right.name),
          )
          for left, right in zip(left_witnesses, right_witnesses)
        )
      )
      violation_body = (
        candidate_relation(
          *head_arguments,
          body_time,
          head_time,
          rank,
          left_lower,
          left_upper,
          *left_witnesses,
        )
        & candidate_relation(
          *head_arguments,
          body_time,
          head_time,
          rank,
          right_lower,
          right_upper,
          *right_witnesses,
        )
        & Filter(expression=ScalarAnd((payload_differs, witness_differs)))
      )
      violation_rule = (violation(rule_index) <= violation_body).named(
        f"{target_name}__aggregate_rank_fd_violation"
      )
      preparation = (*preparation, violation_rule)
      diagnostic_relations = (violation.name,)
    if (
      self.selection.prefer_first_candidate
      and self.selection.initial_lower.value != float32_to_u32(0.0)
    ):
      raise RuleCompileError(
        "first-candidate aggregate fallback currently requires a zero lower initializer"
      )
    initializer_lower = self.selection.initial_lower
    initializer_upper = self.selection.initial_upper
    if self.selection.prefer_first_candidate:
      if self.empty_group_fallback is None:
        raise RuleCompileError(
          "first-candidate aggregate must expose its empty-group fallback return"
        )
      initializer_lower, initializer_upper = self.empty_group_fallback
    initializer_rank = 0xFFFFFFFF if self.selection.prefer_first_candidate else 0
    initializer = (
      proposal(
        *head_arguments,
        body_time,
        head_time,
        initializer_rank,
        initializer_lower.value,
        initializer_upper.value,
      )
      <= body
    ).named(f"{target_name}__aggregate_initializer")
    preparation = (*preparation, initializer)

    selected_rank = Var(f"rule_{rule_index}_selected_rank")
    winner_lower = Var(f"rule_{rule_index}_selected_lower")
    winner_upper = Var(f"rule_{rule_index}_selected_upper")
    selected_proposal = proposal(
      *head_arguments,
      body_time,
      head_time,
      selected_rank,
      winner_lower,
      winner_upper,
    )
    effective_proposal = Relation(
      f"PyReasonCallback{rule_index}EffectiveProposal",
      proposal_arity,
      column_types=(int,) * proposal_arity,
    )
    effective_rank = Var(f"rule_{rule_index}_effective_rank")
    effective_lower = Var(f"rule_{rule_index}_effective_lower")
    effective_upper = Var(f"rule_{rule_index}_effective_upper")
    effective = effective_proposal(
      *head_arguments,
      body_time,
      head_time,
      effective_rank,
      effective_lower,
      effective_upper,
    )
    if self.drop_crossed_interval:
      consistent_winner = (
        effective_proposal(
          *head_arguments,
          body_time,
          head_time,
          selected_rank,
          winner_lower,
          winner_upper,
        )
        <= selected_proposal
        & Filter(
          expression=ScalarCompare(
            "<=",
            ScalarVar(winner_lower.name),
            ScalarVar(winner_upper.name),
          )
        )
      ).named(f"{target_name}__aggregate_effective")
      crossed_fallback = (
        effective_proposal(
          *head_arguments,
          body_time,
          head_time,
          selected_rank,
          float32_to_u32(0.0),
          float32_to_u32(1.0),
        )
        <= selected_proposal
        & Filter(
          expression=ScalarCompare(
            ">",
            ScalarVar(winner_lower.name),
            ScalarVar(winner_upper.name),
          )
        )
      ).named(f"{target_name}__aggregate_crossed_fallback")
      preparation = (*preparation, consistent_winner, crossed_fallback)
    else:
      effective_copy = (
        effective_proposal(
          *head_arguments,
          body_time,
          head_time,
          selected_rank,
          winner_lower,
          winner_upper,
        )
        <= selected_proposal
      ).named(f"{target_name}__aggregate_effective")
      preparation = (*preparation, effective_copy)
    materialization = (
      logical[self.rule.head_predicate](
        *head_arguments,
        head_time,
        effective_lower,
        effective_upper,
      )
      <= effective
    ).named(target_name)
    temporal_events = _temporal_event_rules(
      self.rule,
      rule_index,
      target_name,
      head_time,
      body,
      update_clock,
      horizon_event,
      effective,
      effective_lower,
      effective_upper,
    )
    return CompiledSourceRule(
      materialization=materialization,
      preparation=preparation,
      temporal_events=temporal_events,
      diagnostic_relations=diagnostic_relations,
    )


def _compile_constant_rule(
  rule: Any,
  rule_index: int,
  target_name: str,
  logical: dict[str, Any],
  ticks: dict[int, Any],
  round_sync: Any,
  update_clock: Any,
  horizon_event: Any,
  closed_world_predicates: frozenset[str],
  closed_world_views: dict[str, Any],
  closed_world_eligibility: dict[str, Any],
  head_edge_domain: Any | None,
  witness_relations: WitnessRelations | None,
) -> CompiledSourceRule:
  from srdatalog import Var

  term_vars = _term_variables(rule, rule_index)
  body_time = Var(f"rule_{rule_index}_time")
  head_time = body_time
  clauses: list[Any] = []
  lowers: list[Any] = []
  uppers: list[Any] = []
  bound_terms: set[str] = set()
  for clause_index, clause in enumerate(rule.clauses):
    arguments = tuple(term_vars[term] for term in clause.terms)
    lower = Var(f"rule_{rule_index}_lower_{clause_index}")
    upper = Var(f"rule_{rule_index}_upper_{clause_index}")
    lowers.append(lower)
    uppers.append(upper)
    relation = (
      closed_world_views.get(clause.predicate, logical[clause.predicate])
      if clause.lower == 0.0
      else logical[clause.predicate]
    )
    clauses.extend(
      (
        relation(
          *arguments,
          body_time,
          lower,
          upper,
        ),
        _guard(
          lower,
          upper,
          clause.lower,
          clause.upper,
          closed_world=(clause.predicate in closed_world_predicates and clause.lower == 0.0),
        ),
      )
    )
    if (
      clause.lower == 0.0
      and clause.predicate in closed_world_eligibility
      and all(term not in bound_terms for term in clause.terms)
    ):
      clauses.append(closed_world_eligibility[clause.predicate](*arguments, body_time))
    bound_terms.update(clause.terms)
  clauses.extend((round_sync(body_time), update_clock(body_time)))
  if rule.delay > 0:
    head_time = Var(f"rule_{rule_index}_next_time")
    clauses.append(ticks[rule.delay](body_time, head_time))
  head_arguments = tuple(term_vars[term] for term in rule.head_terms)
  if len(head_arguments) == 2:
    if head_edge_domain is None:
      raise RuleCompileError("binary heads require an active-edge domain relation")
    clauses.append(head_edge_domain(*head_arguments, body_time))
  head_lower = float32_to_u32(rule.head_lower)
  head_upper = float32_to_u32(rule.head_upper)
  body = _conjunction(clauses, rule.name)
  staged_rule = (
    logical[rule.head_predicate](
      *head_arguments,
      head_time,
      head_lower,
      head_upper,
    )
    <= body
  ).named(target_name)
  # PyReason grounds every source rule against one snapshot, then commits all
  # queued heads together.  Callback rules already have two target phases
  # (group candidates, then project the winner).  Give constant-head rules the
  # same producer/commit boundary so one source rule cannot become visible a
  # target iteration earlier merely because it has no user callback.
  materialization = staged_rule.with_grouped_head(
    group_args=(*head_arguments, head_time),
    value_args=(rule_index, head_lower, head_upper),
    value_spec=max_lower_lattice(
      key_columns=tuple(range(len(head_arguments) + 1)),
      rank_column=len(head_arguments) + 1,
      lower_column=len(head_arguments) + 2,
      upper_column=len(head_arguments) + 3,
    ),
  )
  witnesses = _compile_witness_rules(
    witness_relations,
    target_name=target_name,
    head_arguments=head_arguments,
    head_time=head_time,
    rank=rule_index,
    result_lower=head_lower,
    result_upper=head_upper,
    term_vars=term_vars,
    lowers=lowers,
    uppers=uppers,
    body=body,
  )
  event_lower = Var(f"rule_{rule_index}_event_lower")
  event_upper = Var(f"rule_{rule_index}_event_upper")
  activation_source = logical[rule.head_predicate](
    *head_arguments,
    head_time,
    event_lower,
    event_upper,
  )
  temporal_events = _temporal_event_rules(
    rule,
    rule_index,
    target_name,
    head_time,
    body,
    update_clock,
    horizon_event,
    activation_source,
    event_lower,
    event_upper,
  )
  return CompiledSourceRule(
    materialization=materialization,
    temporal_events=temporal_events,
    witnesses=witnesses,
  )


def _temporal_event_rules(
  rule: Any,
  rule_index: int,
  target_name: str,
  head_time: Any,
  body: Any,
  update_clock: Any,
  horizon_event: Any,
  activation_source: Any,
  event_lower: Any,
  event_upper: Any,
) -> tuple[Any, ...]:
  """Separate PyReason's visited horizon from successful-update activation.

  Every delayed grounding extends ``max_rules_time`` even when applying its
  candidate later changes no interval.  Activation reads the grouped,
  interval-merged target rather than an ungrouped callback candidate, so a
  losing aggregate member cannot create a source-rule round.  In the supported
  nonpersistent conflict-free fragment, a consistent informative target starts
  from reset ``[0,1]`` and is therefore a strict update.
  """

  if rule.delay <= 0:
    return ()
  lower_expression = ScalarVar(event_lower.name)
  upper_expression = ScalarVar(event_upper.name)
  informative = Filter(
    expression=ScalarAnd(
      (
        ScalarCompare("<=", lower_expression, upper_expression),
        ScalarOr(
          (
            ScalarCompare(
              ">",
              lower_expression,
              ScalarConst(float32_to_u32(0.0)),
            ),
            ScalarCompare(
              "<",
              upper_expression,
              ScalarConst(float32_to_u32(1.0)),
            ),
          )
        ),
      )
    )
  )
  horizon = (horizon_event(head_time) <= body).named(f"{target_name}__horizon_event")
  activation = (
    update_clock(head_time) <= activation_source & horizon_event(head_time) & informative
  ).named(f"{target_name}__update_event")
  return horizon, activation


def _compile_witness_rules(
  relations: WitnessRelations | None,
  *,
  target_name: str,
  head_arguments: tuple[Any, ...],
  head_time: Any,
  rank: Any,
  result_lower: Any,
  result_upper: Any,
  term_vars: dict[str, Any],
  lowers: list[Any],
  uppers: list[Any],
  body: Any,
) -> tuple[Any, ...]:
  '''Retain only grouped winner occurrences and their supporting groundings.'''
  if relations is None:
    return ()

  witness_arguments = (
    *head_arguments,
    head_time,
    rank,
    result_lower,
    result_upper,
    *term_vars.values(),
    *(item for pair in zip(lowers, uppers) for item in pair),
  )
  winner_arguments = (
    *head_arguments,
    head_time,
    rank,
    result_lower,
    result_upper,
  )
  candidate = (relations.candidate(*witness_arguments) <= body).named(
    f"{target_name}__witness_candidate"
  )
  winner = (relations.winner(*winner_arguments) <= body).named(f"{target_name}__witness_winner")
  history_body = relations.candidate(*witness_arguments) & relations.winner(*winner_arguments)
  history = (relations.history(*witness_arguments) <= history_body).named(
    f"{target_name}__witness_history"
  )
  return candidate, winner, history


def _clause_predicate(clause: Any) -> str:
  return str(clause.predicate if hasattr(clause, "predicate") else clause[0])


def _scalar_variable_names(expression: ScalarExpr) -> frozenset[str]:
  '''Collect endpoint inputs from the monotone min/max scalar fragment.'''
  if isinstance(expression, ScalarVar):
    return frozenset((expression.name,))
  if isinstance(expression, ScalarConst):
    return frozenset()
  if isinstance(expression, (ScalarMin, ScalarMax)):
    return frozenset(
      name for argument in expression.arguments for name in _scalar_variable_names(argument)
    )
  raise RuleCompileError(f"unsupported aggregate endpoint expression {type(expression).__name__}")


def _clause_terms(clause: Any) -> tuple[str, ...]:
  return tuple(clause.terms if hasattr(clause, "terms") else clause[1])


def _term_variables(rule: Any, rule_index: int) -> dict[str, Any]:
  from srdatalog import Var

  terms: list[str] = []
  for term in rule.head_terms:
    if term not in terms:
      terms.append(term)
  for clause in rule.clauses:
    for term in clause.terms:
      if term not in terms:
        terms.append(term)
  return {term: Var(f"rule_{rule_index}_term_{index}") for index, term in enumerate(terms)}


def _guard(
  lower: Any,
  upper: Any,
  lower_bound: float,
  upper_bound: float,
  *,
  closed_world: bool = False,
) -> Filter:
  lower_value = ScalarVar(lower.name)
  upper_value = ScalarVar(upper.name)
  normal = ScalarAnd(
    (
      ScalarCompare("<=", lower_value, upper_value),
      ScalarCompare(
        ">=",
        lower_value,
        ScalarConst(float32_to_u32(lower_bound)),
      ),
      ScalarCompare(
        "<=",
        upper_value,
        ScalarConst(float32_to_u32(upper_bound)),
      ),
    )
  )
  if not closed_world:
    if lower_bound == 0.0 and upper_bound == 1.0:
      crossed_conflict = ScalarCompare(">", lower_value, upper_value)
      return Filter(expression=ScalarOr((normal, crossed_conflict)))
    return Filter(expression=normal)
  default_unknown = ScalarAnd(
    (
      ScalarCompare("==", lower_value, ScalarConst(float32_to_u32(0.0))),
      ScalarCompare("==", upper_value, ScalarConst(float32_to_u32(1.0))),
    )
  )
  crossed_conflict = ScalarCompare(">", lower_value, upper_value)
  return Filter(expression=ScalarOr((normal, default_unknown, crossed_conflict)))


def _conjunction(clauses: list[Any], rule_name: str) -> Any:
  if not clauses:
    raise RuleCompileError(f"rule {rule_name!r} has an empty body; represent it as a fact")
  body = clauses[0]
  for clause in clauses[1:]:
    body = body & clause
  return body
