kernel invalid_constexpr_dynamic_window(
    window: buffer[q15], coefficients: constexpr[q15] = [1, 2, 1]) -> q15:
  return fir(window, coefficients,
             accumulator=exact[40, wrap],
             rounding=nearest_even,
             overflow=saturate)
