def q31_fir_constexpr(
    window: buffer[q31, 4],
    coefficients: constexpr[q31] = [1073741824, -536870912, -536870912, 1073741824]) -> q31:
  return fir(window, coefficients,
             accumulator=exact[64, wrap],
             rounding=nearest_even,
             overflow=saturate)
