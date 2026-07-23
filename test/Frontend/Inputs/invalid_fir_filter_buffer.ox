kernel invalid_fir_filter_buffer(input: buffer[q15], coefficients: buffer[q15]) -> tensor[q15]:
  return fir_filter(input, coefficients, boundary=valid, accumulator=exact[40,saturate], rounding=nearest_even, overflow=saturate)
