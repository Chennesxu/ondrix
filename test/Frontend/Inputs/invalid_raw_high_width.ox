def invalid_raw_high_width(lhs: buffer[q31], rhs: buffer[q31]) -> q31:
  return dot(lhs, rhs, product=raw_high, accumulator=exact[64, saturate], rounding=nearest_even, overflow=saturate)
