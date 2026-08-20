'''Generic isolated CUDA executor for native PyReason rewrite plans.'''

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from srdatalog import CompilerConfig, build_project, compile_jit_project
from srdatalog.ir.codegen.cuda.build.cache import JitProjectLayout
from srdatalog.runtime import (
  cuda_compile_flags,
  cuda_include_paths,
  cuda_libs,
  cuda_link_flags,
  runtime_defines,
  runtime_include_paths,
)
from srdatalog.value_semantics import u32_to_float32

from .model import (
  ExecutionResult,
  NativePlan,
  RuleCandidateRow,
  RuleWitnessOutput,
  TemporalIntervalOutput,
  TemporalIntervalRow,
)

RESULT_PREFIX = 'SRDATALOG_PYREASON_RESULT_JSON='


def _compiler_config(jobs: int) -> CompilerConfig:
  return CompilerConfig(
    include_paths=runtime_include_paths() + cuda_include_paths(),
    defines=runtime_defines(),
    cxx_flags=cuda_compile_flags() + ['-fPIC'],
    link_flags=cuda_link_flags(),
    libs=cuda_libs() + ['boost_container'],
    shared=True,
    jobs=jobs,
  )


def execute(
  plan: NativePlan,
  *,
  cache_base: str | Path,
  jobs: int,
  timeout: float,
) -> ExecutionResult:
  '''Build the target project, isolate CUDA teardown, and decode generic rows.'''
  project = build_project(plan.program, plan.project_name, cache_base=str(cache_base))
  descriptor = {
    'data_dir': str(plan.data_dir.resolve()),
    'jobs': jobs,
    'callback_fd_relations': list(plan.callback_fd_relations),
    'live_time_relation': plan.live_time_relation,
    'outputs': [asdict(output) for output in plan.outputs],
    'project': project,
    'rewriter': plan.rewriter,
    'symbol_names': plan.symbol_names,
    'witness_outputs': [asdict(output) for output in plan.witness_outputs],
  }
  descriptor_path = plan.data_dir / 'srdatalog_pyreason_runtime.json'
  descriptor_path.write_text(json.dumps(descriptor, indent=2) + '\n')
  env = os.environ.copy()
  package_src = str(Path(__file__).resolve().parents[2])
  env['PYTHONPATH'] = os.pathsep.join(
    path for path in (package_src, env.get('PYTHONPATH', '')) if path
  )
  completed = subprocess.run(
    [sys.executable, '-m', 'srdatalog.pyreason.runtime', str(descriptor_path)],
    text=True,
    capture_output=True,
    check=False,
    env=env,
    timeout=timeout,
  )
  if completed.returncode != 0:
    raise RuntimeError(
      f'native SRDatalog PyReason execution failed ({completed.returncode})\n'
      + completed.stdout
      + completed.stderr
    )
  result_line = next(
    (line for line in completed.stdout.splitlines() if line.startswith(RESULT_PREFIX)),
    None,
  )
  if result_line is None:
    raise RuntimeError(
      'native SRDatalog PyReason runner emitted no result\n' + completed.stdout + completed.stderr
    )
  raw = json.loads(result_line.removeprefix(RESULT_PREFIX))
  callback_fd_violations = {
    str(relation): int(count)
    for relation, count in raw.get('callback_fd_violations', {}).items()
    if int(count) > 0
  }
  if callback_fd_violations:
    from .compiler import UnsupportedAnnotationSemantics

    rendered = ', '.join(
      f"{relation}={count}" for relation, count in sorted(callback_fd_violations.items())
    )
    raise UnsupportedAnnotationSemantics(
      "a callback admission rank produced different interval payloads for "
      "one aggregate group; PyReason's ordered lookup needs a logical "
      f"full-grounding rank for this execution ({rendered})"
    )
  rows = tuple(
    TemporalIntervalRow(
      predicate=row['predicate'],
      arguments=tuple(row['arguments']),
      time=int(row['time']),
      lower=float(row['lower']),
      upper=float(row['upper']),
      inconsistent=bool(row.get('inconsistent', False)),
      frozen=bool(row.get('frozen', False)),
      raw_lower=(float(row['raw_lower']) if row.get('raw_lower') is not None else None),
      raw_upper=(float(row['raw_upper']) if row.get('raw_upper') is not None else None),
    )
    for row in raw['rows']
  )
  repaired_callback_inputs = sorted(
    {
      (row.predicate, row.arguments, row.time)
      for row in rows
      if row.inconsistent and row.predicate in plan.callback_read_predicates
    }
  )
  if repaired_callback_inputs:
    from .compiler import UnsupportedAnnotationSemantics

    rendered = ', '.join(
      f"{predicate}{arguments}@{time}"
      for predicate, arguments, time in repaired_callback_inputs[:8]
    )
    raise UnsupportedAnnotationSemantics(
      "a callback reads an inconsistent interval that PyReason repairs to "
      "[0,1] before callback evaluation, while the monotone native world "
      f"retains crossed endpoints ({rendered})"
    )
  if plan.effective_source is not None:
    rule_head_predicates = {rule.head_predicate for rule in plan.effective_source.rules}
    conflict_origins = tuple(row for row in rows if row.inconsistent)
    effective_time = int(raw.get('effective_time', 0) or 0)
    frozen_fact_rewrites = {
      (origin.predicate, origin.arguments, origin.time)
      for origin in conflict_origins
      if any(
        fact.predicate == origin.predicate
        and fact.arguments == origin.arguments
        and (fact.lower != 0.0 or fact.upper != 1.0)
        and max(fact.start_time, origin.time + 1)
        <= min(
          fact.end_time,
          effective_time,
        )
        for fact in plan.effective_source.facts
      )
    }
    hazardous = sorted(
      {
        (row.predicate, row.arguments, row.time)
        for row in conflict_origins
        if row.predicate in rule_head_predicates
      }
      | frozen_fact_rewrites
    )
    if hazardous:
      from .compiler import UnsupportedAnnotationSemantics

      rendered = ', '.join(
        f"{predicate}{arguments}@{time}" for predicate, arguments, time in hazardous[:8]
      )
      raise UnsupportedAnnotationSemantics(
        "a conflict on a rule-head key can be repaired and then overwritten by "
        "a later queued PyReason rule update; that commit order is not a "
        f"commutative interval-lattice join ({rendered})"
      )
  witness_rows = tuple(
    RuleCandidateRow(
      rule_index=int(row['rule_index']),
      rule_name=str(row['rule_name']),
      head_predicate=str(row['head_predicate']),
      head_arguments=tuple(row['head_arguments']),
      head_time=int(row['head_time']),
      rank=(int(row['rank']) if row.get('rank') is not None else None),
      candidate_lower=float(row['candidate_lower']),
      candidate_upper=float(row['candidate_upper']),
      bindings=tuple((str(name), str(value)) for name, value in row['bindings']),
      body_intervals=tuple((float(lower), float(upper)) for lower, upper in row['body_intervals']),
      closed_world_clause_positions=tuple(
        int(position) for position in row.get('closed_world_clause_positions', ())
      ),
    )
    for row in raw.get('witness_rows', ())
  )
  metadata = {key: value for key, value in raw.items() if key not in {'rows', 'witness_rows'}}
  return ExecutionResult(
    rows=rows,
    backend='srdatalog-pyreason',
    rewriter=plan.rewriter,
    metadata=metadata,
    witness_rows=witness_rows,
  )


def _bind(artifact: str) -> Any:
  lib = ctypes.CDLL(artifact, mode=ctypes.RTLD_GLOBAL)
  lib.srdatalog_init.restype = ctypes.c_int
  lib.srdatalog_load_all.argtypes = [ctypes.c_char_p]
  lib.srdatalog_load_all.restype = ctypes.c_int
  lib.srdatalog_run.argtypes = [ctypes.c_ulonglong]
  lib.srdatalog_run.restype = ctypes.c_int
  lib.srdatalog_dev_count.argtypes = [ctypes.c_char_p]
  lib.srdatalog_dev_count.restype = ctypes.c_ulonglong
  lib.srdatalog_dev_ptr.argtypes = [ctypes.c_char_p, ctypes.c_uint]
  lib.srdatalog_dev_ptr.restype = ctypes.c_void_p
  lib.srdatalog_shutdown.restype = ctypes.c_int
  return lib


def _artifact(project: dict[str, object], jobs: int) -> tuple[Path, float]:
  import time

  artifacts = sorted(Path(str(project['dir'])).glob('*.so'))
  if artifacts:
    return artifacts[0], 0.0
  started = time.perf_counter()
  build = compile_jit_project(cast(JitProjectLayout, project), _compiler_config(jobs))
  seconds = time.perf_counter() - started
  if not build.ok():
    diagnostics = [
      result.stderr or result.stdout for result in build.compile_results if result.returncode
    ]
    if build.link_result is not None and build.link_result.returncode:
      diagnostics.append(build.link_result.stderr or build.link_result.stdout)
    raise RuntimeError('\n'.join(diagnostics)[-12000:])
  return Path(build.artifact), seconds


def _copy_columns(
  lib: Any,
  relation: str,
  columns: tuple[int, ...],
) -> tuple[int, dict[int, Any]]:
  count = int(lib.srdatalog_dev_count(relation.encode()))
  if count == 0:
    return count, {}
  result = {}
  copy = lib.cudaMemcpy
  copy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
  copy.restype = ctypes.c_int
  for column in columns:
    pointer = int(lib.srdatalog_dev_ptr(relation.encode(), column))
    host = (ctypes.c_uint32 * count)()
    status = copy(
      ctypes.cast(host, ctypes.c_void_p),
      ctypes.c_void_p(pointer),
      ctypes.sizeof(host),
      2,  # cudaMemcpyDeviceToHost
    )
    if status != 0:
      raise RuntimeError(
        f'cudaMemcpy failed with status {status} while reading {relation}[{column}]'
      )
    result[column] = host
  return count, result


def _decode_rows(
  lib: Any,
  outputs: tuple[TemporalIntervalOutput, ...],
  symbol_names: dict[int, str],
  live_times: frozenset[int] | None = None,
) -> list[dict[str, object]]:
  rows: list[dict[str, object]] = []
  for output in outputs:
    count, columns = _copy_columns(lib, output.relation, output.required_columns)
    for row in range(count):
      time = int(columns[output.time_column][row])
      if live_times is not None and time not in live_times:
        continue
      raw_lower = u32_to_float32(int(columns[output.lower_column][row]))
      raw_upper = u32_to_float32(int(columns[output.upper_column][row]))
      frozen = raw_lower > raw_upper
      # Closed-world range rows are the internal lattice bottom.  They make
      # active-domain joins total during evaluation but are not PyReason facts
      # and must not leak through the public interpretation.  A crossed row is
      # different: it is an inconsistency marker repaired to visible [0,1].
      if output.suppress_bottom and not frozen and raw_lower == 0.0 and raw_upper == 1.0:
        continue
      rows.append(
        {
          'predicate': output.predicate,
          'arguments': [
            symbol_names[int(columns[column][row])] for column in output.argument_columns
          ],
          'time': time,
          # A crossed interval is the monotone internal top value.  PyReason's
          # public repair policy displays top as unknown and freezes it; keep
          # the raw endpoints so demand-driven explanation can seed exactly
          # the conflict keys without carrying witnesses during evaluation.
          'lower': 0.0 if frozen else raw_lower,
          'upper': 1.0 if frozen else raw_upper,
          'frozen': frozen,
          'raw_lower': raw_lower if frozen else None,
          'raw_upper': raw_upper if frozen else None,
        }
      )
  first_conflict: dict[tuple[object, ...], int] = {}
  for output_row in rows:
    if not output_row['frozen']:
      continue
    arguments = cast(list[object], output_row['arguments'])
    key = (output_row['predicate'], *arguments)
    time = int(cast(int, output_row['time']))
    first_conflict[key] = min(first_conflict.get(key, time), time)
  for output_row in rows:
    arguments = cast(list[object], output_row['arguments'])
    key = (output_row['predicate'], *arguments)
    output_row['inconsistent'] = bool(
      output_row['frozen'] and int(cast(int, output_row['time'])) == first_conflict[key]
    )
  return rows


def _decode_witness_rows(
  lib: Any,
  outputs: tuple[RuleWitnessOutput, ...],
  symbol_names: dict[int, str],
  live_times: frozenset[int] | None = None,
) -> list[dict[str, object]]:
  '''Decode logical witness tuples retained by an instrumented replay.'''
  rows: list[dict[str, object]] = []
  for output in outputs:
    count, columns = _copy_columns(lib, output.relation, output.required_columns)
    for row in range(count):
      head_time = int(columns[output.head_time_column][row])
      if live_times is not None and head_time not in live_times:
        continue
      rows.append(
        {
          'rule_index': output.rule_index,
          'rule_name': output.rule_name,
          'head_predicate': output.head_predicate,
          'head_arguments': [
            symbol_names[int(columns[column][row])] for column in output.head_argument_columns
          ],
          'head_time': head_time,
          'rank': (
            int(columns[output.rank_column][row]) if output.rank_column is not None else None
          ),
          'candidate_lower': u32_to_float32(int(columns[output.candidate_lower_column][row])),
          'candidate_upper': u32_to_float32(int(columns[output.candidate_upper_column][row])),
          'bindings': [
            [name, symbol_names[int(columns[column][row])]]
            for name, column in output.binding_columns
          ],
          'body_intervals': [
            [
              u32_to_float32(int(columns[lower_column][row])),
              u32_to_float32(int(columns[upper_column][row])),
            ]
            for lower_column, upper_column in zip(
              output.body_lower_columns,
              output.body_upper_columns,
            )
          ],
          'closed_world_clause_positions': list(output.closed_world_clause_positions),
        }
      )
  return rows


def _run_descriptor(path: str | Path) -> None:
  import time

  descriptor = json.loads(Path(path).read_text())
  outputs = tuple(TemporalIntervalOutput(**output) for output in descriptor['outputs'])
  witness_outputs = tuple(
    RuleWitnessOutput(**output) for output in descriptor.get('witness_outputs', ())
  )
  symbols = {int(identifier): symbol for identifier, symbol in descriptor['symbol_names'].items()}
  artifact, compile_seconds = _artifact(descriptor['project'], int(descriptor['jobs']))
  lib = _bind(str(artifact.resolve()))
  if lib.srdatalog_init() != 0:
    raise RuntimeError('srdatalog_init failed')
  started = time.perf_counter()
  if lib.srdatalog_load_all(str(descriptor['data_dir']).encode()) != 0:
    raise RuntimeError('srdatalog_load_all failed')
  load_seconds = time.perf_counter() - started
  started = time.perf_counter()
  if lib.srdatalog_run(0) != 0:
    raise RuntimeError('srdatalog_run failed')
  run_seconds = time.perf_counter() - started
  live_time_relation = descriptor.get('live_time_relation')
  live_times: frozenset[int] | None = None
  if live_time_relation is not None:
    live_count, live_columns = _copy_columns(
      lib,
      str(live_time_relation),
      (0,),
    )
    live_times = frozenset(int(live_columns[0][row]) for row in range(live_count))
  result = {
    'callback_fd_violations': {
      str(relation): int(lib.srdatalog_dev_count(str(relation).encode()))
      for relation in descriptor.get('callback_fd_relations', ())
    },
    'compile_seconds': compile_seconds,
    'load_seconds': load_seconds,
    'rewriter': descriptor['rewriter'],
    'effective_time': max(live_times, default=0) if live_times is not None else None,
    'rows': _decode_rows(lib, outputs, symbols, live_times),
    'run_seconds': run_seconds,
    'witness_rows': _decode_witness_rows(
      lib,
      witness_outputs,
      symbols,
      live_times,
    ),
  }
  print(RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)
  lib.srdatalog_shutdown()
  sys.stdout.flush()
  sys.stderr.flush()
  os._exit(0)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('descriptor')
  args = parser.parse_args()
  _run_descriptor(args.descriptor)


if __name__ == '__main__':
  main()
