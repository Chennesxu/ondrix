def invalid_fir_filter_full_overflow(input: tensor[q15,9223372036854775807], coefficients: tensor[q15,2]) -> tensor[q15,1]:
  return fir_filter(input, coefficients, boundary=full, accumulator=exact[40,saturate], rounding=nearest_even, overflow=saturate)
