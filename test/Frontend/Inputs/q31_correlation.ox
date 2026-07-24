def q31_correlation(input: tensor[q31,6], kernel: tensor[q31,3]) -> tensor[q31,4]:
  return correlation(input, kernel, accumulator=exact[64,wrap], rounding=toward_zero, overflow=wrap)
