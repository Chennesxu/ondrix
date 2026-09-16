def invalid_raw_high_q15(lhs: buffer[q15], rhs: buffer[q15]) -> q15:
  return dot(lhs, rhs, product=raw_high, accumulator=exact[40, saturate], rounding=nearest_even, overflow=saturate)
