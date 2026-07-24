def invalid_constexpr_q15_range(
    window: buffer[q15, 3], coefficients: constexpr[q15] = [1, 32768, 1]) -> q15:
  return fir(window, coefficients,
             accumulator=exact[40, wrap],
             rounding=nearest_even,
             overflow=saturate)
