from dataclasses import replace

import pytest

from srdatalog.pyreason import (
  ArgMaxLower,
  ClauseLower,
  ClauseUpper,
  EndpointMinimum,
  GroupedAnnotation,
  WitnessInterval,
  annotation_semantics,
  get_annotation_semantics,
  grouped_argmax_lower_of_minimum,
  register_annotation_semantics,
)


def test_callable_carries_declarative_grouped_annotation() -> None:
  semantics = grouped_argmax_lower_of_minimum(0, 2, rank_clause=2)

  @annotation_semantics(semantics)
  def aggregate(*_args: object) -> tuple[float, float]:
    return 1.0, 1.0

  assert get_annotation_semantics(aggregate) is semantics
  assert semantics == GroupedAnnotation(
    witness=WitnessInterval(
      lower=EndpointMinimum((ClauseLower(0), ClauseLower(2))),
      upper=EndpointMinimum((ClauseUpper(0), ClauseUpper(2))),
    ),
    aggregate=ArgMaxLower(rank_clause=2),
  )


def test_registration_is_idempotent_but_rejects_conflicts() -> None:
  def aggregate(*_args: object) -> tuple[float, float]:
    return 1.0, 1.0

  first = grouped_argmax_lower_of_minimum(0, rank_clause=0)
  register_annotation_semantics(aggregate, first)
  register_annotation_semantics(aggregate, first)

  with pytest.raises(ValueError, match='different SRDatalog semantics'):
    register_annotation_semantics(
      aggregate,
      replace(first, aggregate=ArgMaxLower(rank_clause=1)),
    )


def test_grouped_minimum_requires_value_clause() -> None:
  with pytest.raises(ValueError, match='at least one value clause'):
    grouped_argmax_lower_of_minimum(rank_clause=0)
