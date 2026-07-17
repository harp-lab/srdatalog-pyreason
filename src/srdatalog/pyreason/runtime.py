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
    'outputs': [asdict(output) for output in plan.outputs],
    'project': project,
    'rewriter': plan.rewriter,
    'symbol_names': plan.symbol_names,
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
    (
      line
      for line in completed.stdout.splitlines()
      if line.startswith(RESULT_PREFIX)
    ),
    None,
  )
  if result_line is None:
    raise RuntimeError(
      'native SRDatalog PyReason runner emitted no result\n'
      + completed.stdout
      + completed.stderr
    )
  raw = json.loads(result_line.removeprefix(RESULT_PREFIX))
  rows = tuple(
    TemporalIntervalRow(
      predicate=row['predicate'],
      arguments=tuple(row['arguments']),
      time=int(row['time']),
      lower=float(row['lower']),
      upper=float(row['upper']),
    )
    for row in raw['rows']
  )
  metadata = {key: value for key, value in raw.items() if key != 'rows'}
  return ExecutionResult(
    rows=rows,
    backend='srdatalog-pyreason',
    rewriter=plan.rewriter,
    metadata=metadata,
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
      result.stderr or result.stdout
      for result in build.compile_results
      if result.returncode
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
) -> list[dict[str, object]]:
  rows: list[dict[str, object]] = []
  for output in outputs:
    count, columns = _copy_columns(lib, output.relation, output.required_columns)
    for row in range(count):
      rows.append(
        {
          'predicate': output.predicate,
          'arguments': [
            symbol_names[int(columns[column][row])]
            for column in output.argument_columns
          ],
          'time': int(columns[output.time_column][row]),
          'lower': u32_to_float32(int(columns[output.lower_column][row])),
          'upper': u32_to_float32(int(columns[output.upper_column][row])),
        }
      )
  return rows


def _run_descriptor(path: str | Path) -> None:
  import time

  descriptor = json.loads(Path(path).read_text())
  outputs = tuple(TemporalIntervalOutput(**output) for output in descriptor['outputs'])
  symbols = {
    int(identifier): symbol
    for identifier, symbol in descriptor['symbol_names'].items()
  }
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
  result = {
    'compile_seconds': compile_seconds,
    'load_seconds': load_seconds,
    'rewriter': descriptor['rewriter'],
    'rows': _decode_rows(lib, outputs, symbols),
    'run_seconds': run_seconds,
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
