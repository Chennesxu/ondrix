def q15_fir_constexpr(
    window: buffer[q15, 5],
    coefficients: constexpr[q15] = [16384, -8192, 4096, -8192, 16384]) -> q15:
  return fir(window, coefficients,
             accumulator=exact[40, wrap],
             rounding=nearest_even,
             overflow=saturate)
