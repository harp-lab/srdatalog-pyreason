import ctypes

from srdatalog.pyreason.runtime import _copy_columns


class _Copy:
  argtypes = None
  restype = None

  def __call__(self, destination, source, size, kind):
    assert kind == 2
    ctypes.memmove(destination, source, size)
    return 0


class _Library:
  def __init__(self) -> None:
    self.values = (ctypes.c_uint32 * 3)(7, 11, 13)
    self.cudaMemcpy = _Copy()

  def srdatalog_dev_count(self, relation: bytes) -> int:
    assert relation == b'Relation'
    return 3

  def srdatalog_dev_ptr(self, relation: bytes, column: int) -> int:
    assert relation == b'Relation'
    assert column == 2
    return ctypes.addressof(self.values)


def test_copy_columns_needs_no_array_library() -> None:
  count, columns = _copy_columns(_Library(), 'Relation', (2,))

  assert count == 3
  assert list(columns[2]) == [7, 11, 13]
