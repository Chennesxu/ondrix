kernel invalid_constexpr_dot(
    input: buffer[q15, 3], coefficients: constexpr[q15] = [1, 2, 1]) -> q15:
  return dot(input, coefficients,
             accumulator=exact[40, wrap],
             rounding=nearest_even,
             overflow=saturate)
