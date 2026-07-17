'''Declarative, application-neutral semantics for PyReason annotations.'''

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias, TypeVar


@dataclass(frozen=True)
class ClauseLower:
  clause: int


@dataclass(frozen=True)
class ClauseUpper:
  clause: int


@dataclass(frozen=True)
class EndpointConstant:
  value: float


@dataclass(frozen=True)
class EndpointMinimum:
  terms: tuple[EndpointExpression, ...]

  def __post_init__(self) -> None:
    if not self.terms:
      raise ValueError('endpoint minimum requires at least one term')


@dataclass(frozen=True)
class EndpointMaximum:
  terms: tuple[EndpointExpression, ...]

  def __post_init__(self) -> None:
    if not self.terms:
      raise ValueError('endpoint maximum requires at least one term')


EndpointExpression: TypeAlias = (
  ClauseLower | ClauseUpper | EndpointConstant | EndpointMinimum | EndpointMaximum
)


@dataclass(frozen=True)
class WitnessInterval:
  lower: EndpointExpression
  upper: EndpointExpression


@dataclass(frozen=True)
class ArgMaxLower:
  '''Select the witness with greatest lower bound and minimum stable rank.'''

  rank_clause: int


@dataclass(frozen=True)
class GroupedAnnotation:
  '''Map every body-join witness to an interval, then reduce one head group.'''

  witness: WitnessInterval
  aggregate: ArgMaxLower


AnnotationSemantics: TypeAlias = GroupedAnnotation

_SEMANTICS_ATTRIBUTE = '__srdatalog_annotation_semantics__'
_AnnotationFunction = TypeVar('_AnnotationFunction')


def register_annotation_semantics(
  annotation_function: object,
  semantics: AnnotationSemantics,
) -> None:
  '''Attach compiler-readable semantics to a first-class Python callable.'''
  existing = getattr(annotation_function, _SEMANTICS_ATTRIBUTE, None)
  if existing is not None and existing != semantics:
    raise ValueError('annotation function already has different SRDatalog semantics')
  try:
    setattr(annotation_function, _SEMANTICS_ATTRIBUTE, semantics)
  except (AttributeError, TypeError) as exc:
    raise TypeError('annotation callable cannot carry SRDatalog semantics') from exc


def annotation_semantics(
  semantics: AnnotationSemantics,
) -> Callable[[_AnnotationFunction], _AnnotationFunction]:
  '''Decorate a Python annotation callback with its declarative semantics.'''

  def decorate(annotation_function: _AnnotationFunction) -> _AnnotationFunction:
    register_annotation_semantics(annotation_function, semantics)
    return annotation_function

  return decorate


def get_annotation_semantics(
  annotation_function: object,
) -> AnnotationSemantics | None:
  value = getattr(annotation_function, _SEMANTICS_ATTRIBUTE, None)
  return value if isinstance(value, GroupedAnnotation) else None


def grouped_argmax_lower_of_minimum(
  *clauses: int,
  rank_clause: int,
) -> GroupedAnnotation:
  '''Common grouped aggregate: endpoint-wise minimum, then ARG MAX lower.'''
  if not clauses:
    raise ValueError('grouped annotation requires at least one value clause')
  lower_terms = tuple(ClauseLower(clause) for clause in clauses)
  upper_terms = tuple(ClauseUpper(clause) for clause in clauses)
  return GroupedAnnotation(
    witness=WitnessInterval(
      lower=EndpointMinimum(lower_terms),
      upper=EndpointMinimum(upper_terms),
    ),
    aggregate=ArgMaxLower(rank_clause=rank_clause),
  )
