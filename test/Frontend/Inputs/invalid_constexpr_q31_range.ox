kernel invalid_constexpr_q31_range(
    window: buffer[q31, 3], coefficients: constexpr[q31] = [1, 2147483648, 1]) -> q31:
  return fir(window, coefficients,
             accumulator=exact[64, wrap],
             rounding=nearest_even,
             overflow=saturate)
