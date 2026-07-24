def q31_fir_filter_full(input: tensor[q31,4], coefficients: tensor[q31,3]) -> tensor[q31,6]:
  return fir_filter(input, coefficients, boundary=full, accumulator=exact[64,wrap], rounding=toward_zero, overflow=wrap)
