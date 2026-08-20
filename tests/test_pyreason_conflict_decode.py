import json
import subprocess

import pytest

from srdatalog.pyreason import (
  NativePlan,
  RuleWitnessOutput,
  TemporalIntervalOutput,
  UnsupportedAnnotationSemantics,
)
from srdatalog.pyreason.runtime import (
  RESULT_PREFIX,
  _decode_rows,
  _decode_witness_rows,
  execute,
)
from srdatalog.value_semantics import float32_to_u32


class _Column:
  def __init__(self, values):
    self.values = values

  def __getitem__(self, row):
    return self.values[row]


def test_callback_rank_payload_violation_is_rejected_before_decode(
  tmp_path,
  monkeypatch,
) -> None:
  monkeypatch.setattr(
    'srdatalog.pyreason.runtime.build_project',
    lambda *_args, **_kwargs: 'mock-project',
  )
  payload = {
    'callback_fd_violations': {
      'PyReasonCallback0RankFunctionalViolation': 1,
    },
  }
  completed = subprocess.CompletedProcess(
    args=[],
    returncode=0,
    stdout=RESULT_PREFIX + json.dumps(payload) + '\n',
    stderr='',
  )
  monkeypatch.setattr(
    'srdatalog.pyreason.runtime.subprocess.run',
    lambda *_args, **_kwargs: completed,
  )
  plan = NativePlan(
    program=object(),
    project_name='callback-fd-test',
    data_dir=tmp_path,
    outputs=(),
    symbol_names={},
    rewriter='test',
    callback_fd_relations=('PyReasonCallback0RankFunctionalViolation',),
  )

  with pytest.raises(
    UnsupportedAnnotationSemantics,
    match='admission rank produced different interval payloads',
  ):
    execute(plan, cache_base=tmp_path / 'cache', jobs=1, timeout=1)

  descriptor = json.loads((tmp_path / 'srdatalog_pyreason_runtime.json').read_text())
  assert descriptor['callback_fd_relations'] == ['PyReasonCallback0RankFunctionalViolation']


def test_callback_read_of_repaired_conflict_is_rejected(
  tmp_path,
  monkeypatch,
) -> None:
  monkeypatch.setattr(
    'srdatalog.pyreason.runtime.build_project',
    lambda *_args, **_kwargs: 'mock-project',
  )
  payload = {
    'rows': [
      {
        'predicate': 'input',
        'arguments': ['a'],
        'time': 0,
        'lower': 0.0,
        'upper': 1.0,
        'inconsistent': True,
        'frozen': True,
        'raw_lower': 0.8,
        'raw_upper': 0.2,
      }
    ],
    'callback_fd_violations': {},
  }
  completed = subprocess.CompletedProcess(
    args=[],
    returncode=0,
    stdout=RESULT_PREFIX + json.dumps(payload) + '\n',
    stderr='',
  )
  monkeypatch.setattr(
    'srdatalog.pyreason.runtime.subprocess.run',
    lambda *_args, **_kwargs: completed,
  )
  plan = NativePlan(
    program=object(),
    project_name='callback-conflict-test',
    data_dir=tmp_path,
    outputs=(),
    symbol_names={},
    rewriter='test',
    callback_read_predicates=('input',),
  )

  with pytest.raises(
    UnsupportedAnnotationSemantics,
    match=r'PyReason repairs to \[0,1\] before callback evaluation',
  ):
    execute(plan, cache_base=tmp_path / 'cache', jobs=1, timeout=1)


def test_crossed_internal_interval_decodes_as_repaired_conflict(monkeypatch) -> None:
  output = TemporalIntervalOutput(
    relation='R',
    predicate='p',
    argument_columns=(0,),
    time_column=1,
    lower_column=2,
    upper_column=3,
  )
  columns = {
    0: _Column([7]),
    1: _Column([2]),
    2: _Column([float32_to_u32(0.8)]),
    3: _Column([float32_to_u32(0.2)]),
  }
  monkeypatch.setattr(
    'srdatalog.pyreason.runtime._copy_columns',
    lambda _lib, _relation, _columns: (1, columns),
  )

  assert _decode_rows(object(), (output,), {7: 'a'}) == [
    {
      'predicate': 'p',
      'arguments': ['a'],
      'time': 2,
      'lower': 0.0,
      'upper': 1.0,
      'inconsistent': True,
      'frozen': True,
      'raw_lower': 0.800000011920929,
      'raw_upper': 0.20000000298023224,
    }
  ]


def test_bottom_is_suppressed_only_for_a_materialized_cwa_range(monkeypatch) -> None:
  visible = TemporalIntervalOutput('Visible', 'visible', (0,), 1, 2, 3)
  internal = TemporalIntervalOutput(
    'Internal',
    'internal',
    (0,),
    1,
    2,
    3,
    suppress_bottom=True,
  )
  columns = {
    0: _Column([7]),
    1: _Column([0]),
    2: _Column([float32_to_u32(0.0)]),
    3: _Column([float32_to_u32(1.0)]),
  }
  monkeypatch.setattr(
    'srdatalog.pyreason.runtime._copy_columns',
    lambda _lib, _relation, _columns: (1, columns),
  )

  assert [row['predicate'] for row in _decode_rows(object(), (visible, internal), {7: 'a'})] == [
    'visible'
  ]


def test_rows_after_the_derived_live_horizon_are_not_public(monkeypatch) -> None:
  output = TemporalIntervalOutput('Visible', 'visible', (0,), 1, 2, 3)
  columns = {
    0: _Column([7, 7]),
    1: _Column([0, 1]),
    2: _Column([float32_to_u32(1.0), float32_to_u32(0.0)]),
    3: _Column([float32_to_u32(1.0), float32_to_u32(1.0)]),
  }
  monkeypatch.setattr(
    'srdatalog.pyreason.runtime._copy_columns',
    lambda _lib, _relation, _columns: (2, columns),
  )

  assert [row['time'] for row in _decode_rows(object(), (output,), {7: 'a'}, frozenset({0}))] == [0]


def test_demanded_witness_decodes_logical_identity_and_body_versions(monkeypatch) -> None:
  output = RuleWitnessOutput(
    relation='Witness',
    rule_index=3,
    rule_name='derive-result',
    head_predicate='result',
    head_argument_columns=(0,),
    head_time_column=1,
    rank_column=2,
    candidate_lower_column=3,
    candidate_upper_column=4,
    binding_columns=(('X', 5),),
    body_lower_columns=(6,),
    body_upper_columns=(7,),
    closed_world_clause_positions=(0,),
  )
  columns = {
    0: _Column([7]),
    1: _Column([2]),
    2: _Column([11]),
    3: _Column([float32_to_u32(1.0)]),
    4: _Column([float32_to_u32(1.0)]),
    5: _Column([7]),
    6: _Column([float32_to_u32(0.0)]),
    7: _Column([float32_to_u32(1.0)]),
  }
  monkeypatch.setattr(
    'srdatalog.pyreason.runtime._copy_columns',
    lambda _lib, _relation, _columns: (1, columns),
  )

  assert _decode_witness_rows(object(), (output,), {7: 'a'}) == [
    {
      'rule_index': 3,
      'rule_name': 'derive-result',
      'head_predicate': 'result',
      'head_arguments': ['a'],
      'head_time': 2,
      'rank': 11,
      'candidate_lower': 1.0,
      'candidate_upper': 1.0,
      'bindings': [['X', 'a']],
      'body_intervals': [[0.0, 1.0]],
      'closed_world_clause_positions': [0],
    }
  ]
