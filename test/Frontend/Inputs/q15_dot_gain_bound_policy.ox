def q15_dot_gain_bound_policy(lhs: buffer[q15,8], rhs: buffer[q15,8]) -> q15:
  return dot(lhs, rhs, gain_bound=3, accumulator=exact[40, saturate], rounding=nearest_even, overflow=saturate)
