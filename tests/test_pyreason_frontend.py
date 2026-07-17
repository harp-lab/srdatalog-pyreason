from srdatalog.pyreason import SourceProgram, SourceRule


def annotation(annotations, weights):
  return annotations[0][0].lower, annotations[0][0].upper


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
