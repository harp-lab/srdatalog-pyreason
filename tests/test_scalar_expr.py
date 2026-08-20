import math

import srdatalog.ir.mir.types as mir
from srdatalog import (
  Program,
  Relation,
  Var,
  compile_to_mir,
  float32_to_u32,
  u32_to_float32,
)
from srdatalog.dsl import (
  Filter,
  Let,
  ScalarCompare,
  ScalarConst,
  ScalarFloat32Sub,
  ScalarMin,
  ScalarOr,
  ScalarVar,
  render_scalar,
  scalar_dependencies,
)


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


def test_float32_bit_subtraction_decodes_before_arithmetic() -> None:
  expression = ScalarFloat32Sub(ScalarConst(0x3F800000), ScalarVar('upper'))

  assert scalar_dependencies(expression) == ('upper',)
  assert render_scalar(expression) == (
    '__float_as_uint(__uint_as_float(1065353216) - __uint_as_float(upper))'
  )


def test_scalar_or_tracks_all_dependencies() -> None:
  expression = ScalarOr(
    (
      ScalarCompare('==', ScalarVar('lower'), ScalarConst(0)),
      ScalarCompare('>', ScalarVar('lower'), ScalarVar('upper')),
    )
  )

  assert scalar_dependencies(expression) == ('lower', 'upper')
  assert render_scalar(expression) == '(lower == 0) || (lower > upper)'


def test_probability_zero_has_one_canonical_unsigned_encoding() -> None:
  assert float32_to_u32(0.0) == float32_to_u32(-0.0) == 0
  assert math.copysign(1.0, u32_to_float32(0x80000000)) == 1.0
