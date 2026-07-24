def q31_fir_filter_valid(input: tensor[q31,6], coefficients: tensor[q31,3]) -> tensor[q31,4]:
  return fir_filter(input, coefficients, boundary=valid, accumulator=exact[64,wrap], rounding=toward_zero, overflow=wrap)
