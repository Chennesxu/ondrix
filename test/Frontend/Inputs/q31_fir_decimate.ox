def q31_fir_decimate(
    input: tensor[q31,12], coefficients: tensor[q31,5]) -> tensor[q31,4]:
  return fir_decimate(input, coefficients, factor=2, accumulator=exact[64,saturate], rounding=nearest_even, overflow=saturate)
