def invalid_auto_dynamic(
    lhs: buffer[q15], rhs: buffer[q15]) -> q15:
  return dot(lhs, rhs)
