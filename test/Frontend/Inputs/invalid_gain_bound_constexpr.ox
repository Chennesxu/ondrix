def q15_bad(lhs: buffer[q15,4], rhs: constexpr[q15] = [32767, 32767, 2, 0]) -> q15:
  return dot(lhs, rhs, gain_bound=2)
