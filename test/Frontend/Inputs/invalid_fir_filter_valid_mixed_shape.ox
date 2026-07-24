def invalid_fir_filter_valid_mixed_shape(input: tensor[q15], coefficients: tensor[q15,3]) -> tensor[q15,4]:
  return fir_filter(input, coefficients, boundary=valid, accumulator=exact[40,saturate], rounding=nearest_even, overflow=saturate)
