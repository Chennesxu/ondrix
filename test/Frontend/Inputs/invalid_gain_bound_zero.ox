def q15_bad(lhs: buffer[q15,8], rhs: buffer[q15,8]) -> q15:
  return dot(lhs, rhs, gain_bound=0)
