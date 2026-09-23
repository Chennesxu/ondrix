def q15_dot_gain_bound(lhs: buffer[q15,16], rhs: buffer[q15,16]) -> q15:
  return dot(lhs, rhs, gain_bound=2)
