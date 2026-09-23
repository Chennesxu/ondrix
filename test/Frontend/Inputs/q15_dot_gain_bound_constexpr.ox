def q15_dot_gain_bound_constexpr(lhs: buffer[q15,4], rhs: constexpr[q15] = [32767, 32767, 1, 0]) -> q15:
  return dot(lhs, rhs, gain_bound=2)
