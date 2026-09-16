def invalid_product_selection(lhs: buffer[q31], rhs: buffer[q31]) -> q31:
  return dot(lhs, rhs, product=low, accumulator=exact[64, saturate], rounding=nearest_even, overflow=saturate)
