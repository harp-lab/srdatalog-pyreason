import srdatalog.ir.mir.types as mir
from srdatalog import Program, Relation, Var, compile_to_mir
from srdatalog.dsl import Filter, Let, ScalarCompare, ScalarConst, ScalarMin, ScalarVar


def test_typed_scalar_expression_lowers_once_at_mir_boundary() -> None:
  key, lower, upper, result = (Var(name) for name in ('key', 'lower', 'upper', 'result'))
  source = Relation('ScalarSource', 3)
  target = Relation('ScalarTarget', 3)
  rule = (
    target(key, result, upper)
    <= source(key, lower, upper)
    & Let(
      result.name,
      expression=ScalarMin((ScalarVar(lower.name), ScalarConst(7))),
    )
    & Filter(
      expression=ScalarCompare(
        '<=',
        ScalarVar(result.name),
        ScalarVar(upper.name),
      )
    )
  ).named('TypedScalar')

  lowered = compile_to_mir(Program(rules=[rule]), apply_mir_passes=False)
  plan = next(step for step, _ in lowered.steps if isinstance(step, mir.FixpointPlan))
  pipeline = next(op for op in plan.instructions if isinstance(op, mir.ExecutePipeline))
  binding = next(op for op in pipeline.pipeline if isinstance(op, mir.ConstantBind))
  predicate = next(op for op in pipeline.pipeline if isinstance(op, mir.Filter))

  assert binding.code == 'std::min(lower, 7)'
  assert binding.deps == ['lower']
  assert predicate.code == 'return result <= upper;'
  assert predicate.vars == ['result', 'upper']
