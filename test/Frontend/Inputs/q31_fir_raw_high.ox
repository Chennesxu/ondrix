def q31_fir_raw_high(
    window: buffer[q31, 4],
    coefficients: constexpr[q31] = [1073741824, -536870912, 2147483647, -2147483648]) -> q31:
  return fir(window, coefficients,
             product=raw_high,
             accumulator=exact[40, wrap],
             rounding=nearest_ties_positive,
             overflow=wrap)
